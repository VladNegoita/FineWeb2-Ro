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
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import defaultdict


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


class BertClassifier(nn.Module):
    def __init__(
        self, model_name: str, classes_num: int, extra_layer_sizes=[], dropout_rate=0.1, finetune: bool = False,
    ):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.layers = nn.ModuleList()
        if not finetune:
            self.name = (
                f"{model_name.split('/')[-1]}_{'_'.join(map(str, extra_layer_sizes))}"
            )
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            self.name = f"{model_name.split('/')[-1]}_finetune_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = True

        prev_size = self.bert.config.hidden_size
        for size in extra_layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.layers.append(nn.Linear(prev_size, classes_num))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        x = pooled_output
        for layer in self.layers:
            x = layer(x)

        return x

    def model_unique_name(self) -> str:
        return self.name


class TextClassifierDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

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
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_model(
    model, device, train_loader, val_loader, n_epochs, learning_rate, bert_learning_rate
):
    print("\n--- Training Started ---")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "train_f1_weighted": [],
        "val_f1_weighted": [],
        "train_classification_report": [],
        "val_classification_report": [],
        "train_confusion_matrix": [],
        "val_confusion_matrix": [],
    }

    loss_fn = nn.CrossEntropyLoss()
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
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc="Training", leave=False)

        all_train_preds = []
        all_train_labels = []

        for batch in train_progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": loss.item()})

            all_train_preds.extend(outputs.detach().cpu())
            all_train_labels.extend(labels.detach().cpu())

        avg_train_loss = train_loss / len(train_loader)

        all_train_preds_tensor = torch.stack(all_train_preds)
        all_train_labels_tensor = torch.stack(all_train_labels)
        train_pred_classes = torch.argmax(all_train_preds_tensor, dim=1)
        train_true_classes = all_train_labels_tensor
        train_accuracy = (train_pred_classes == train_true_classes).float().mean().item()
        train_f1_macro = f1_score(
            train_true_classes.numpy(),
            train_pred_classes.numpy(),
            average="macro",
            zero_division=0,
        )
        train_f1_weighted = f1_score(
            train_true_classes.numpy(),
            train_pred_classes.numpy(),
            average="weighted",
            zero_division=0,
        )

        history["train_loss"].append(avg_train_loss)
        history["train_f1_macro"].append(train_f1_macro)
        history["train_f1_weighted"].append(train_f1_weighted)
        history["train_accuracy"].append(train_accuracy)
        history["train_classification_report"].append(
            classification_report(
                train_true_classes.numpy(),
                train_pred_classes.numpy(),
                output_dict=True,
                zero_division=0,
                digits=4,
            )
        )
        history["train_confusion_matrix"].append(
            confusion_matrix(
                train_true_classes.numpy(),
                train_pred_classes.numpy(),
                labels=list(SECONDARY_TASKS[label_name].values()),
            ).tolist()
        )

        print(f"Average Training Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Train F1 Macro: {train_f1_macro:.4f} | Train F1 Weighted: {train_f1_weighted:.4f}")
        print()

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                val_progress_bar.set_postfix({"loss": loss.item()})

                all_val_preds.extend(outputs.detach().cpu())
                all_val_labels.extend(labels.detach().cpu())

        avg_val_loss = val_loss / len(val_loader)
        

        all_val_preds_tensor = torch.stack(all_val_preds)
        all_val_labels_tensor = torch.stack(all_val_labels)

        val_pred_classes = torch.argmax(all_val_preds_tensor, dim=1)
        val_true_classes = all_val_labels_tensor

        val_accuracy = (val_pred_classes == val_true_classes).float().mean().item()
        val_f1_macro = f1_score(
            val_true_classes.numpy(),
            val_pred_classes.numpy(),
            average="macro",
            zero_division=0,
        )

        val_f1_weighted = f1_score(
            val_true_classes.numpy(),
            val_pred_classes.numpy(),
            average="weighted",
            zero_division=0,
        )

        history["val_accuracy"].append(val_accuracy)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_f1_weighted"].append(val_f1_weighted)
        history["val_loss"].append(avg_val_loss)
        history["val_classification_report"].append(
            classification_report(
                val_true_classes.numpy(),
                val_pred_classes.numpy(),
                output_dict=True,
                zero_division=0,
                digits=4,
            )
        )
        history["val_confusion_matrix"].append(
            confusion_matrix(
                val_true_classes.numpy(),
                val_pred_classes.numpy(),
                labels=list(SECONDARY_TASKS[label_name].values()),
            ).tolist()
        )
        print(f"Average Validation Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}")
        print()



    print("\n--- Training Complete ---")

    return history


def load_dataset_split(path, train_limit, label_name):
    if label_name not in SECONDARY_TASKS:
        raise ValueError(f"Invalid class name: {label_name}. Must be one of {list(SECONDARY_TASKS.keys())}.")
    print(f"Loading dataset from {path} with labels as class: {label_name}...")

    shard_reader = ShardLoader(path)

    train_examples = shard_reader.get_examples(shard_reader.get_train_indices())
    val_examples = shard_reader.get_examples(shard_reader.get_val_indices())

    print(f"Number of training examples: {len(train_examples)}")
    print(f"Number of validation examples: {len(val_examples)}")

    def filter_correct_examples(examples):
        return [
            example
            for example in examples
            if example[label_name] in SECONDARY_TASKS[label_name]
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

    class_mapper = SECONDARY_TASKS[label_name]
    def extract_text_and_labels(examples):

        return [example["text"] for example in examples], [
            int(class_mapper[example[label_name]]) for example in examples
        ]

    train_texts, train_labels = extract_text_and_labels(train_examples)
    val_texts, val_labels = extract_text_and_labels(val_examples)
    return train_texts, train_labels, val_texts, val_labels


def main(
    base_model_name: str,
    extra_layer_sizes: List[int],
    finetune: bool,
    bert_learning_rate: float,
    lr: float,
    epochs: int,
    train_limit: int,
    label_name: str,
):
    print(f"Base model name: {base_model_name}")
    print(f"Extra layer sizes: {extra_layer_sizes}")
    print(f"Finetune: {finetune}")
    print(f"BERT learning rate: {bert_learning_rate}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Training limit: {train_limit}")
    print(f"Label name: {label_name}")
    print()

    train_texts, train_labels, val_texts, val_labels = load_dataset_split(
        "/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results", train_limit, label_name
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = TextClassifierDataset(
        train_texts, train_labels, tokenizer, 512
    )  # Standard BERT max length

    val_dataset = TextClassifierDataset(val_texts, val_labels, tokenizer, 512)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertClassifier(
        base_model_name,
        classes_num=len(SECONDARY_TASKS[label_name]),
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
    )
    with open(
        f"training/results/history_{model.model_unique_name()}_config_{label_name}_{lr}_{bert_learning_rate}_{epochs}_{train_limit}.json",
        "w",
    ) as f:
        json.dump(history, f, indent=4)

    # torch.save(model.state_dict(), f"training/models/{model.model_unique_name()}.pth")
    # print(f"Model saved as {model.model_unique_name()}.pth")


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
        "--label_name",
        type=str,
        default="topic",
        choices=list(SECONDARY_TASKS.keys()),
        help="The secondary task to train on. Must be one of: " + ", ".join(SECONDARY_TASKS.keys()),
    )

    args = parser.parse_args()
    extra_layer_sizes = args.extra_layers if args.extra_layers else []
    base_model_name = args.model_name
    finetune = args.finetune
    bert_learning_rate = args.bert_lr
    lr = args.lr
    epochs = args.epochs
    train_limit = args.train_limit
    label_name = args.label_name
    main(
        base_model_name,
        extra_layer_sizes,
        finetune,
        bert_learning_rate,
        lr,
        epochs,
        train_limit,
        label_name,
    )
