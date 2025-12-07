from tkinter import TOP
from matplotlib.pyplot import hist
from regex import T
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from shard_loader import ShardLoader
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import json
from argparse import ArgumentParser
from typing import List
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


class BertMultiTask(nn.Module):
    def __init__(
        self, model_name, extra_layer_sizes=[], dropout_rate=0.1, finetune: bool = False
    ):
        super(BertMultiTask, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.layers = nn.ModuleList()
        if not finetune:
            self.name = f"{model_name.split('/')[-1]}_multitask_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            self.name = f"{model_name.split('/')[-1]}_finetune_multitask_{'_'.join(map(str, extra_layer_sizes))}"
            for param in self.bert.parameters():
                param.requires_grad = True

        prev_size = self.bert.config.hidden_size
        for size in extra_layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.reg_head = nn.Linear(prev_size, 1)
        self.classification_head = nn.Linear(prev_size, len(TOPICS))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        x = pooled_output
        for layer in self.layers:
            x = layer(x)

        reg_output = self.reg_head(x).squeeze(-1)
        class_output = self.classification_head(x)
        return reg_output, class_output

    def model_unique_name(self) -> str:
        return self.name


class TextMultiTaskDataset(Dataset):
    def __init__(self, texts, reg_labels, class_labels, tokenizer, max_length):
        self.texts = texts
        self.reg_labels = reg_labels
        self.class_labels = class_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        reg_label = self.reg_labels[idx]
        class_label = self.class_labels[idx]

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
            "class_labels": torch.tensor(class_label, dtype=torch.long),
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
        "train_loss": [],
        "val_loss": [],
        "train_reg_predictions": [],
        "val_reg_predictions": [],
        "train_class_predictions": [],
        "val_class_predictions": [],
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
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc="Training", leave=False)

        all_train_reg_preds = []
        all_train_reg_labels = []
        all_train_class_preds = []
        all_train_class_labels = []

        for batch in train_progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reg_labels = batch["reg_labels"].to(device)
            class_labels = batch["class_labels"].to(device)

            optimizer.zero_grad()
            reg_preds, class_preds = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss_reg = mse_loss(reg_preds, reg_labels)
            loss_class = ce_loss(class_preds, class_labels)
            loss = loss_reg + alpha * loss_class

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": loss.item()})

            all_train_reg_preds.extend(reg_preds.detach().cpu())
            all_train_reg_labels.extend(reg_labels.detach().cpu())
            all_train_class_preds.extend(class_preds.detach().cpu())
            all_train_class_labels.extend(class_labels.detach().cpu())

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_reg_predictions"].append(
            torch.tensor(all_train_reg_preds).tolist()
        )
        history["train_class_predictions"].append(
            [torch.tensor(x).clone().detach().tolist() for x in all_train_class_preds]
        )
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        all_val_reg_preds = []
        all_val_reg_labels = []
        all_val_class_preds = []
        all_val_class_labels = []

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                reg_labels = batch["reg_labels"].to(device)
                class_labels = batch["class_labels"].to(device)

                reg_preds, class_preds = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                reg_loss = mse_loss(reg_preds, reg_labels)
                class_loss = ce_loss(class_preds, class_labels)
                loss = reg_loss + alpha * class_loss

                val_loss += loss.item()
                val_progress_bar.set_postfix({"loss": loss.item()})

                all_val_reg_preds.extend(reg_preds.detach().cpu())
                all_val_reg_labels.extend(reg_labels.detach().cpu())
                all_val_class_preds.extend(class_preds.detach().cpu())
                all_val_class_labels.extend(class_labels.detach().cpu())

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        history["val_reg_predictions"].append(torch.tensor(all_val_reg_preds).tolist())
        history["val_class_predictions"].append(
            [torch.tensor(x).clone().detach().tolist() for x in all_val_class_preds]
        )

    history["train_reg_labels"] = torch.tensor(all_train_reg_labels).tolist()
    history["val_reg_labels"] = torch.tensor(all_val_reg_labels).tolist()
    history["train_class_labels"] = torch.tensor(all_train_class_labels).tolist()
    history["val_class_labels"] = torch.tensor(all_val_class_labels).tolist()

    print("\n--- Training Complete ---")

    return history


def load_dataset_split(path, train_limit):
    shard_reader = ShardLoader(path)

    train_examples = shard_reader.get_examples(shard_reader.get_train_indices())
    val_examples = shard_reader.get_examples(shard_reader.get_val_indices())

    print(f"Number of training examples: {len(train_examples)}")
    print(f"Number of validation examples: {len(val_examples)}")

    def filter_correct_examples(examples, key="int_score", key2="topic"):
        return [
            example
            for example in examples
            if isinstance(example[key], int)
            and example[key] >= 0
            and example[key] <= 5
            and example[key2] in TOPIC_TO_ID
        ]

    train_examples = filter_correct_examples(train_examples)
    val_examples = filter_correct_examples(val_examples)

    if len(train_examples) > train_limit:
        print(
            f"Truncated training examples to {train_limit} from {len(train_examples)}."
        )
        train_examples = train_examples[:train_limit]

    print(f"Filtered number of training examples: {len(train_examples)}")
    print(f"Filtered number of validation examples: {len(val_examples)}")
    print()

    def extract_text_and_labels(
        examples, text_key="text", label_key="int_score", label_key_2="topic"
    ):
        return (
            [example[text_key] for example in examples],
            [int(example[label_key]) for example in examples],
            [int(TOPIC_TO_ID[example[label_key_2]]) for example in examples],
        )

    train_texts, train_reg_labels, train_class_labels = extract_text_and_labels(
        train_examples
    )
    val_texts, val_reg_labels, val_class_labels = extract_text_and_labels(val_examples)
    return (
        train_texts,
        train_reg_labels,
        train_class_labels,
        val_texts,
        val_reg_labels,
        val_class_labels,
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
    print()

    (
        train_texts,
        train_reg_labels,
        train_class_labels,
        val_texts,
        val_reg_labels,
        val_class_labels,
    ) = load_dataset_split(
        "/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results", train_limit
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = TextMultiTaskDataset(
        train_texts, train_reg_labels, train_class_labels, tokenizer, 512
    )  # Standard BERT max length

    val_dataset = TextMultiTaskDataset(
        val_texts, val_reg_labels, val_class_labels, tokenizer, 512
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

    with open(
        f"training/results/history_{model.model_unique_name()}_config_{lr}_{bert_learning_rate}_{epochs}_{train_limit}.json",
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
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for the classification loss in the combined loss function.",
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
