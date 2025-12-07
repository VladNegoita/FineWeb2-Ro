import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from model import BertMultiTask
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
from scipy.stats import pearsonr, spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_NAME = 'readerbench/RoBERT-base'
EXTRA_LAYERS = [256]
STATE_DICT_PATH = "/export/home/acs/stud/v/vlad_andrei.negoita/app/project/training/models/RoBERT-base_finetune_all_tasks_256_config_0.0001_3e-06_3_4000000_0.8.pth"


class TextMultiTaskDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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
        }


def load_human_dataset():
    print("Loading human dataset...", flush=True)
    ds = load_dataset("JQL-AI/JQL-Human-Edu-Annotations", split="test")
    rom = ds.filter(lambda x: x['metadata']['language'] == 'ro')
    texts = [x['text'] for x in rom]
    scores = [float(x['score']) for x in rom]
    print(f"Loaded {len(texts)} texts and scores from human dataset.", flush=True)
    return texts, scores


def load_llm_dataset():
    print("Loading LLM dataset...", flush=True)
    ds = load_dataset("JQL-AI/JQL-LLM-Edu-Annotations", split="train")
    rom = ds.filter(lambda x: x['language'] == 'ron')
    texts = [x['text'] for x in rom]
    for example in rom:
        llms = list(example['edu_score'].keys())
        break

    assert len(llms) == 3, "Expected exactly 3 LLM predictions in the dataset"
    print(f"Found LLMs: {', '.join(llms)}", flush=True)

    scores = {llm: [] for llm in llms + ['average']}
    for example in rom:
        for llm in llms:
            scores[llm].append(float(example['edu_score'][llm]))
        scores['average'].append(sum(float(example['edu_score'][llm]) for llm in llms) / len(llms))
    print(f"Loaded {len(texts)} texts and scores from LLM dataset.", flush=True)
    return texts, scores


def load_model(device, state_dict_path=STATE_DICT_PATH):
    state_dict = torch.load(state_dict_path, map_location=device)
    model = BertMultiTask(
        model_name=MODEL_NAME,
        extra_layer_sizes=EXTRA_LAYERS,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_small_model(texts):
    device = torch.device("cuda")
    model = load_model(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = TextMultiTaskDataset(texts, tokenizer, 512)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_reg_preds = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reg_preds, _ = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            reg_preds = reg_preds.cpu().numpy().tolist()
            all_reg_preds.extend(reg_preds)

            print(f"Processed batch {idx + 1}/{len(loader)}", flush=True)
    
    print(f"Finished processing", flush=True)

    assert len(all_reg_preds) == len(texts), "Mismatch in number of predictions and texts"
    return all_reg_preds


def compare_predictions(texts, scores, label):
    print(f"Comparing predictions for {label} dataset...")

    preds = infer_small_model(texts)
    assert len(scores) == len(preds), "Mismatch in number of scores and predictions"

    rmse = root_mean_squared_error(scores, preds)
    mae = mean_absolute_error(scores, preds)
    pearson_corr, _ = pearsonr(scores, preds)
    spearman_corr, _ = spearmanr(scores, preds)

    print(f"Dataset size: {len(scores)}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

    preds = np.round(preds)
    scores = np.round(scores) # not needed, but for consistency

    cm = confusion_matrix(scores, preds)
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Annotated')
    plt.title('Confusion Matrix')
    plt.savefig(f'photos/confusion_matrix_comparison_{label}.png')

    print(f"Classification report:\n{classification_report(scores, preds, zero_division=0)}")
    print("\n\n")


def main():
    human_texts, human_scores = load_human_dataset()
    llm_texts, llm_scores = load_llm_dataset()
    compare_predictions(human_texts, human_scores, "human")

    for llm, scores in llm_scores.items():
        print(f"Comparing predictions for LLM: {llm}")
        compare_predictions(llm_texts, scores, llm)


if __name__ == "__main__":
    main()
