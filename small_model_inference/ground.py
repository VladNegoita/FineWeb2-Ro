import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from model import BertMultiTask
import json
from jql.src.utils.regression_head import RegressionHead
from jql.src.utils.embedder import get_embedder_instance
from transformers.utils.hub import cached_file
from torch.amp import autocast

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


def load_jql_human_dataset():
    print("Loading JQL human dataset...", flush=True)
    ds = load_dataset("JQL-AI/JQL-Human-Edu-Annotations", split="test")
    rom = ds.filter(lambda x: x['metadata']['language'] == 'ro')
    texts = [x['text'] for x in rom]
    scores = [float(x['score']) for x in rom]
    print(f"Loaded {len(texts)} texts and scores from JQL human dataset.", flush=True)
    return texts, scores


def load_human_dataset():
    print("Loading ours human dataset...", flush=True)
    with open("datasets/small_dataset.json", "r") as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    scores = [item['int_score'] for item in data]
    return texts, scores


def load_test_set():
    print("Loading test set...", flush=True)
    data = []
    for test_shard_index in range(1, 17):
        file_path = f"/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results/fineweb2_{test_shard_index}_results.json"
        with open(file_path, "r") as f:
            data.extend(json.load(f))

    texts = [item['text'] for item in data]
    scores = [item['int_score'] for item in data]
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


def infer_jql(texts):
    device = torch.device(f"cuda")
    base_model = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', device, torch.bfloat16)

    regression_head_checkpoints = {
        'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
        'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
        'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
    }

    regression_heads = {}
    for name, path in regression_head_checkpoints.items():
        regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(torch.bfloat16)

    scores = {name: [] for name in regression_heads.keys()}
    BATCH_SIZE = 16
    for start_idx in range(0, len(texts), BATCH_SIZE):
        batch = texts[start_idx:start_idx + BATCH_SIZE]
        with torch.no_grad(), autocast("cuda"):
            embeddings = base_model.embed(batch)
            outputs = dict()
            for name, regression_head in regression_heads.items():
                outputs[name] = regression_head(embeddings).cpu().squeeze(1)
            for i, _ in enumerate(batch):
                for name in outputs:
                    scores[name].append(float(outputs[name][i].item()))

    return scores


def main():
    human_texts, human_scores = load_human_dataset()
    test_texts, test_scores = load_test_set()
    jql_texts, jql_scores = load_jql_human_dataset()

    results = []
    for dataset_name, texts, scores in [
        ("Human", human_texts, human_scores),
        ("Test", test_texts, test_scores),
        ("JQL Human", jql_texts, jql_scores)
    ]:
        print(f"Processing {dataset_name} dataset...", flush=True)
        our_preds = infer_small_model(texts)
        jql_preds = infer_jql(texts)
        
        results.append({
            "dataset": dataset_name,
            "int_scores": scores,
            "reg_preds": our_preds,
            "jql_preds": jql_preds
        })

    with open(f"ro_vs_eng/jql_ground.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved.", flush=True)


if __name__ == "__main__":
    main()
