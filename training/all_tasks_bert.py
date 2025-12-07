import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from shard_loader import ShardLoader
import torch.optim as optim
from tqdm.auto import tqdm
import json
from argparse import ArgumentParser
from typing import List
from collections import defaultdict
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    f1_score,
    confusion_matrix,
    classification_report,
)

from scipy.stats import pearsonr, spearmanr

TOPICS = [
    "Conținut pentru adulți",
    "Artă și design",
    "Dezvoltare software",
    "Crime și investigații",
    "Educație și joburi",
    "Electronică și hardware",
    "Divertisment",
    "Viață socială",
    "Modă și frumusețe",
    "Finanțe și afaceri",
    "Mâncare și băuturi",
    "Jocuri",
    "Sănătate",
    "Istorie și geografie",
    "Hobby-uri și casă",
    "Industrial",
    "Literatură",
    "Politică",
    "Religie",
    "Știință, matematică și tehnologie",
    "Software",
    "Sport și fitness",
    "Transport",
    "Turism și călătorii",
]

TOPIC_TO_ID = defaultdict(lambda: "ERROR")
for i, topic in enumerate(TOPICS):
    TOPIC_TO_ID[topic] = i

FORMATS = [
    "Articol academic",
    "Cuprins",
    "Scriere creativă",
    "Pagină de asistență pentru clienți",
    "Forum de discuții",
    "Întrebări frecvente (FAQs)",
    "Conținut incomplet",
    "Articol de cunoștințe",
    "Notificări legale",
    "Articol de tip listă",
    "Articol de știri",
    "Scriere non-ficțiune",
    "Pagină despre organizație",
    "Anunț organizațional",
    "Pagină personală",
    "Blog personal",
    "Pagină de produs",
    "Forum întrebări și răspunsuri",
    "Spam și reclame",
    "Date structurate",
    "Scriere tehnică",
    "Transcriere sau interviu",
    "Tutorial sau ghid",
    "Recenzii ale utilizatorilor",
]

FORMAT_TO_ID = defaultdict(lambda: "ERROR")
for i, format in enumerate(FORMATS):
    FORMAT_TO_ID[format] = i

AGE_GROUPS = [
    "Preșcolar",
    "Școală primară",
    "Școală gimnazială",
    "Liceu",
    "Licență",
    "Post-universitar",
]

AGE_GROUP_TO_ID = defaultdict(lambda: "ERROR")
for i, age_group in enumerate(AGE_GROUPS):
    AGE_GROUP_TO_ID[age_group] = i

SECONDARY_TASKS = {
    "topic": TOPIC_TO_ID,
    "format": FORMAT_TO_ID,
    "age_group": AGE_GROUP_TO_ID,
}

SECONDARY_TASKS_REVERSED = {
    "topic": TOPICS,
    "format": FORMATS,
    "age_group": AGE_GROUPS,
}


class BertMultiTask(nn.Module):
    def __init__(
        self, model_name, extra_layer_sizes=[], dropout_rate=0.1, finetune: bool = False
    ):
        super(BertMultiTask, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.layers = nn.ModuleList()
        if not finetune:
            self.name = f"{model_name.split('/')[-1]}_all_tasks_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            self.name = f"{model_name.split('/')[-1]}_finetune_all_tasks_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = True

        prev_size = self.bert.config.hidden_size
        for size in extra_layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.reg_head = nn.Linear(prev_size, 1)  # for education value

        self.classification_heads = nn.ModuleDict()
        for task_name, id_map in SECONDARY_TASKS.items():
            self.classification_heads[task_name] = nn.Linear(prev_size, len(id_map))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        x = pooled_output
        for layer in self.layers:
            x = layer(x)

        reg_output = self.reg_head(x).squeeze(-1)
        classes_outputs = {}
        for task, head in self.classification_heads.items():
            classes_outputs[task] = head(x)

        return reg_output, classes_outputs

    def model_unique_name(self) -> str:
        return self.name


class TextMultiTaskDataset(Dataset):
    def __init__(self, texts, reg_labels, classes_labels, tokenizer, max_length):
        self.texts = texts
        self.reg_labels = reg_labels
        self.classes_labels = classes_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        reg_label = self.reg_labels[idx]
        classes_label = self.classes_labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "reg_labels": torch.tensor(reg_label, dtype=torch.float),
            "classes_labels": {
                task_name: torch.tensor(label, dtype=torch.long)
                for task_name, label in classes_label.items()
            },
        }


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    n_epochs,
    learning_rate,
    bert_learning_rate,
    alpha,
):
    print("\n--- Training Started ---")
    
    history = {
        # General
        "train_loss": [],
        "val_loss": [],
    
        # Regression
        "train_reg_loss": [],
        "val_reg_loss": [],

        "train_reg_rmse": [],
        "val_reg_rmse": [],

        "train_reg_mae": [],
        "val_reg_mae": [],

        "train_reg_pearson": [],
        "val_reg_pearson": [],

        "train_reg_spearman": [],
        "val_reg_spearman": [],
        
        "train_reg_f1_macro": [],
        "val_reg_f1_macro": [],

        "train_reg_f1_weighted": [],
        "val_reg_f1_weighted": [],

        "train_reg_confusion_matrix": [],
        "val_reg_confusion_matrix": [],

        "train_reg_classification_report": [],
        "val_reg_classification_report": [],

        # Classification
        "train_classes_loss": [],
        "val_classes_loss": [],

        "train_classes_f1_macro": [],
        "val_classes_f1_macro": [],

        "train_classes_f1_weighted": [],
        "val_classes_f1_weighted": [],

        "train_classes_confusion_matrix": [],
        "val_classes_confusion_matrix": [],

        "train_classes_classification_report": [],
        "val_classes_classification_report": [],
    }

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [
            {"params": model.layers.parameters(), "lr": learning_rate},
            {"params": model.bert.parameters(), "lr": bert_learning_rate},
        ]
    )

    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")

        model.train()
        train_loss, total_reg_loss, total_classes_loss = 0.0, 0.0, {task: 0.0 for task in SECONDARY_TASKS.keys()}
        train_progress_bar = tqdm(train_loader, desc="Training", leave=False)

        all_train_reg_preds = []
        all_train_reg_labels = []

        all_train_classes_preds = {task: [] for task in SECONDARY_TASKS.keys()}
        all_train_classes_labels = {task: [] for task in SECONDARY_TASKS.keys()}

        for batch in train_progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reg_labels = batch["reg_labels"].to(device)
            classes_labels = {
                task: labels.to(device)
                for task, labels in batch["classes_labels"].items()
            }

            optimizer.zero_grad()
            reg_preds, classes_preds = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss_classes = {
                task: ce_loss(classes_preds[task], classes_labels[task])
                for task in classes_preds
            }
            reg_loss = mse_loss(reg_preds, reg_labels)

            loss = reg_loss  # regression loss
            for task, task_loss in loss_classes.items():
                loss += alpha * task_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            total_reg_loss += reg_loss.item()
            for task, task_loss in loss_classes.items():
                total_classes_loss[task] += task_loss.item()

            train_progress_bar.set_postfix({"loss": loss.item()})

            all_train_reg_preds.extend(reg_preds.detach().cpu())
            all_train_reg_labels.extend(reg_labels.detach().cpu())
            for task in classes_preds.keys():
                all_train_classes_preds[task].extend(classes_preds[task].detach().cpu())
                all_train_classes_labels[task].extend(classes_labels[task].detach().cpu())


        avg_train_loss = train_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        avg_classes_loss = {task: total_loss / len(train_loader) for task, total_loss in total_classes_loss.items()}
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Losses
        history["train_loss"].append(avg_train_loss)
        history["train_reg_loss"].append(avg_reg_loss)
        history["train_classes_loss"].append(avg_classes_loss)

        # Regression
        train_reg_preds_np = torch.tensor(all_train_reg_preds).detach().cpu().numpy().reshape(-1, 1).flatten()
        train_reg_labels_np = torch.tensor(all_train_reg_labels).detach().cpu().numpy().reshape(-1, 1).flatten()

        history["train_reg_rmse"].append(float(root_mean_squared_error(train_reg_labels_np, train_reg_preds_np)))
        history["train_reg_mae"].append(float(mean_absolute_error(train_reg_labels_np, train_reg_preds_np)))
        history["train_reg_pearson"].append(float(pearsonr(train_reg_labels_np, train_reg_preds_np).correlation))
        history["train_reg_spearman"].append(float(spearmanr(train_reg_labels_np, train_reg_preds_np).correlation))

        train_reg_preds_np_rounded = train_reg_preds_np.clip(0, 5).round().astype(int)
        train_reg_labels_np_rounded = train_reg_labels_np.clip(0, 5).round().astype(int)

        history["train_reg_f1_macro"].append(float(
            f1_score(train_reg_labels_np_rounded, train_reg_preds_np_rounded, average="macro", zero_division=0)
        ))
        history["train_reg_f1_weighted"].append(float(
            f1_score(train_reg_labels_np_rounded, train_reg_preds_np_rounded, average="weighted", zero_division=0)
        ))

        history["train_reg_confusion_matrix"].append(
            confusion_matrix(train_reg_labels_np_rounded, train_reg_preds_np_rounded).tolist()
        )
        history["train_reg_classification_report"].append(
            classification_report(
                train_reg_labels_np_rounded,
                train_reg_preds_np_rounded,
                output_dict=True,
                zero_division=0,
                digits=4,
            )
        )

        # Classification
        train_classes_preds_tensor = {task: torch.stack(all_train_classes_preds[task]) for task in all_train_classes_preds.keys()}
        train_classes_labels_tensor = {task: torch.stack(all_train_classes_labels[task]) for task in all_train_classes_labels.keys()}

        train_classes_preds = {task: torch.argmax(preds, dim=1).detach().cpu().numpy() for task, preds in train_classes_preds_tensor.items()}
        train_classes_labels = {task: labels.detach().cpu().numpy() for task, labels in train_classes_labels_tensor.items()}

        history["train_classes_f1_macro"].append({task : float(
            f1_score(
                train_classes_labels[task],
                train_classes_preds[task],
                average="macro",
                zero_division=0,
            )
        ) for task in train_classes_preds.keys()})

        history["train_classes_f1_weighted"].append({task : float(
            f1_score(
                train_classes_labels[task],
                train_classes_preds[task],
                average="weighted",
                zero_division=0,
            )
        )for task in train_classes_preds.keys()})

        history["train_classes_confusion_matrix"].append(
            {task: confusion_matrix(train_classes_labels[task], train_classes_preds[task]).tolist() for task in train_classes_preds.keys()}
        )

        history["train_classes_classification_report"].append(
            {task: classification_report(
                train_classes_labels[task],
                train_classes_preds[task],
                output_dict=True,
                zero_division=0,
                digits=4,
            ) for task in train_classes_preds.keys()}
        )

        model.eval()
        val_loss, val_reg_loss, val_classes_loss = 0.0, 0.0, {task: 0.0 for task in SECONDARY_TASKS.keys()}
        val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        all_val_reg_preds = []
        all_val_reg_labels = []

        all_val_classes_preds = {task: [] for task in SECONDARY_TASKS.keys()}
        all_val_classes_labels = {task: [] for task in SECONDARY_TASKS.keys()}

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                reg_labels = batch["reg_labels"].to(device)
                classes_labels = {
                    task: label.to(device)
                    for task, label in batch["classes_labels"].items()
                }

                reg_preds, classes_preds = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                loss_classes = {
                    task: ce_loss(classes_preds[task], classes_labels[task])
                    for task in classes_preds
                }
                reg_loss = mse_loss(reg_preds, reg_labels)

                loss = reg_loss  # start with regression loss
                for task, task_loss in loss_classes.items():
                    loss += alpha * task_loss

                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                for task, task_loss in loss_classes.items():
                    val_classes_loss[task] += task_loss.item()

                val_progress_bar.set_postfix({"loss": loss.item()})

                all_val_reg_preds.extend(reg_preds.detach().cpu())
                all_val_reg_labels.extend(reg_labels.detach().cpu())

                for task in classes_preds.keys():
                    all_val_classes_preds[task].extend(classes_preds[task].detach().cpu())
                    all_val_classes_labels[task].extend(classes_labels[task].detach().cpu())


        avg_val_loss = val_loss / len(val_loader)
        avg_val_reg_loss = val_reg_loss / len(val_loader)
        avg_val_classes_loss = {
            task: total_loss / len(val_loader)
            for task, total_loss in val_classes_loss.items()
        }

        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Losses
        history["val_loss"].append(avg_val_loss)
        history["val_reg_loss"].append(avg_val_reg_loss)
        history["val_classes_loss"].append(avg_val_classes_loss)

        # Regression
        val_reg_preds_np = torch.tensor(all_val_reg_preds).detach().cpu().numpy().reshape(-1, 1).flatten()
        val_reg_labels_np = torch.tensor(all_val_reg_labels).detach().cpu().numpy().reshape(-1, 1).flatten()

        history["val_reg_rmse"].append(float(root_mean_squared_error(val_reg_labels_np, val_reg_preds_np)))
        history["val_reg_mae"].append(float(mean_absolute_error(val_reg_labels_np, val_reg_preds_np)))
        history["val_reg_pearson"].append(float(pearsonr(val_reg_labels_np, val_reg_preds_np).correlation))
        history["val_reg_spearman"].append(float(spearmanr(val_reg_labels_np, val_reg_preds_np).correlation))

        val_reg_preds_np_rounded = val_reg_preds_np.clip(0, 5).round().astype(int)
        val_reg_labels_np_rounded = val_reg_labels_np.clip(0, 5).round().astype(int)

        history["val_reg_f1_macro"].append(float(
            f1_score(val_reg_labels_np_rounded, val_reg_preds_np_rounded, average="macro", zero_division=0)
        ))
        history["val_reg_f1_weighted"].append(float(
            f1_score(val_reg_labels_np_rounded, val_reg_preds_np_rounded, average="weighted", zero_division=0)
        ))
        history["val_reg_confusion_matrix"].append(
            confusion_matrix(val_reg_labels_np_rounded, val_reg_preds_np_rounded).tolist()
        )
        history["val_reg_classification_report"].append(
            classification_report(
                val_reg_labels_np_rounded,
                val_reg_preds_np_rounded,
                output_dict=True,
                zero_division=0,
                digits=4,
            )
        )

        # Classification
        val_classes_preds_tensor = {task: torch.stack(all_val_classes_preds[task]) for task in all_val_classes_preds.keys()}
        val_classes_labels_tensor = {task: torch.stack(all_val_classes_labels[task]) for task in all_val_classes_labels.keys()}

        val_classes_preds = {task: torch.argmax(preds, dim=1).detach().cpu().numpy() for task, preds in val_classes_preds_tensor.items()}
        val_classes_labels = {task: labels.detach().cpu().numpy() for task, labels in val_classes_labels_tensor.items()}

        history["val_classes_f1_macro"].append({task : float(
            f1_score(
                val_classes_labels[task],
                val_classes_preds[task],
                average="macro",
                zero_division=0,
            )
        ) for task in val_classes_preds.keys()})

        history["val_classes_f1_weighted"].append({task : float(
            f1_score(
                val_classes_labels[task],
                val_classes_preds[task],
                average="weighted",
                zero_division=0,
            )
        )for task in val_classes_preds.keys()})

        history["val_classes_confusion_matrix"].append(
            {task: confusion_matrix(val_classes_labels[task], val_classes_preds[task]).tolist() for task in val_classes_preds.keys()}
        )

        history["val_classes_classification_report"].append(
            {task: classification_report(
                val_classes_labels[task],
                val_classes_preds[task],
                output_dict=True,
                zero_division=0,
                digits=4,
            ) for task in val_classes_preds.keys()}
        )


    print("\n--- Training Complete ---")
    return history


def load_dataset_split(path, train_limit):
    shard_reader = ShardLoader(path)

    train_examples = shard_reader.get_examples(shard_reader.get_train_indices())
    val_examples = shard_reader.get_examples(shard_reader.get_val_indices())

    print(f"Number of training examples: {len(train_examples)}")
    print(f"Number of validation examples: {len(val_examples)}")

    def filter_correct_examples(examples):
        return [
            example
            for example in examples
            if isinstance(example["int_score"], int)
            and example["int_score"] >= 0
            and example["int_score"] <= 5
            and example["topic"] in TOPIC_TO_ID
            and example["format"] in FORMAT_TO_ID
            and example["age_group"] in AGE_GROUP_TO_ID
        ]

    train_examples = filter_correct_examples(train_examples)
    val_examples = filter_correct_examples(val_examples)

    print(f"Filtered number of training examples: {len(train_examples)}")
    print(f"Filtered number of validation examples: {len(val_examples)}")
    print()

    if len(train_examples) > train_limit:
        print(
            f"Truncated training examples to {train_limit} from {len(train_examples)}."
        )
        train_examples = train_examples[:train_limit]

    def extract_text_and_labels(examples):
        return (
            [example["text"] for example in examples],
            [int(example["int_score"]) for example in examples],
            [
                {
                    task_name: int(task_to_id[example[task_name]])
                    for task_name, task_to_id in SECONDARY_TASKS.items()
                }
                for example in examples
            ],
        )

    train_texts, train_reg_labels, train_classes_labels = extract_text_and_labels(
        train_examples
    )
    val_texts, val_reg_labels, val_classes_labels = extract_text_and_labels(
        val_examples
    )
    return (
        train_texts,
        train_reg_labels,
        train_classes_labels,
        val_texts,
        val_reg_labels,
        val_classes_labels,
    )


def main(
    base_model_name: str,
    extra_layer_sizes: List[int],
    finetune: bool,
    bert_learning_rate: float,
    lr: float,
    epochs: int,
    train_limit: int,
    alpha: float,
):
    print(f"Base model name: {base_model_name}")
    print(f"Extra layer sizes: {extra_layer_sizes}")
    print(f"Finetune: {finetune}")
    print(f"BERT learning rate: {bert_learning_rate}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Training limit: {train_limit}")
    print(f"Alpha: {alpha}")
    print()

    (
        train_texts,
        train_reg_labels,
        train_classes_labels,
        val_texts,
        val_reg_labels,
        val_classes_labels,
    ) = load_dataset_split(
        "/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results", train_limit
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = TextMultiTaskDataset(
        train_texts, train_reg_labels, train_classes_labels, tokenizer, 512
    )  # Standard BERT max length

    val_dataset = TextMultiTaskDataset(
        val_texts, val_reg_labels, val_classes_labels, tokenizer, 512
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertMultiTask(
        base_model_name,
        extra_layer_sizes=extra_layer_sizes,
        dropout_rate=0.1,
        finetune=finetune,
    )
    model.to(device)

    print(f"Model: {model.model_unique_name()}")
    history = train_model(
        model,
        device,
        train_loader,
        val_loader,
        n_epochs=epochs,
        learning_rate=lr,
        bert_learning_rate=bert_learning_rate,
        alpha=alpha,
    )

    full_config_name = f"{model.model_unique_name()}_config_{lr}_{bert_learning_rate}_{epochs}_{train_limit}_{alpha}"
    with open(
        f"training/results/history_{full_config_name}.json",
        "w",
    ) as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), f"training/models/{full_config_name}.pth")
    print(f"Model saved as {full_config_name}.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Pre-trained model name from Hugging Face Hub",
        required=True,
    )

    parser.add_argument(
        "--extra_layers",
        type=int,
        nargs="*",
        help="(Optional) A list of integers (provide them space-separated, e.g., --int-list 10 20 30) that represent the sizes of the extra layers to be added to the model. If not provided, no extra layers will be added.",
    )

    parser.add_argument(
        "--finetune",
        action="store_true",
        help="(Optional) If provided, the base model will be fine-tuned.",
    )

    parser.add_argument(
        "--bert_lr",
        type=float,
        default=3e-5,
        help="Learning rate for BERT model. Useful when finetuning.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for the rest of the model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for training."
    )
    parser.add_argument(
        "--train_limit", type=int, default=40000, help="Limit training examples."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for the secondary losses in the combined loss function.",
    )

    args = parser.parse_args()
    extra_layer_sizes = args.extra_layers if args.extra_layers else []
    base_model_name = args.model_name
    finetune = args.finetune
    bert_learning_rate = args.bert_lr
    lr = args.lr
    epochs = args.epochs
    train_limit = args.train_limit
    alpha = args.alpha
    main(
        base_model_name,
        extra_layer_sizes,
        finetune,
        bert_learning_rate,
        lr,
        epochs,
        train_limit,
        alpha,
    )