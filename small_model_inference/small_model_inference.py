import torch
import argparse
import torch.multiprocessing as mp
from shard_manager import ShardManager
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from taxonomy import SECONDARY_TASKS, SECONDARY_TASKS_REVERSED
from model import BertMultiTask

MODEL_NAME = 'readerbench/RoBERT-base'
EXTRA_LAYERS = [256]
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

def infer_shard(model, shard_number: int, device: torch.device):
    examples = ShardManager.load_shard(shard_number)
    texts = [example["text"] for example in examples]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = TextMultiTaskDataset(texts, tokenizer, 512)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_reg_preds = []
    all_classes_preds = {task: [] for task in SECONDARY_TASKS.keys()}

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            reg_preds, classes_preds = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            reg_preds = reg_preds.cpu().numpy().tolist()
            all_reg_preds.extend(reg_preds)

            for task in classes_preds.keys():
                all_classes_preds[task].extend(classes_preds[task].detach().cpu())

            print(f"Processed batch {idx + 1}/{len(loader)} from shard {shard_number}", flush=True)
    
    print(f"Finished processing shard {shard_number}", flush=True)
    classes_preds_tensor = {task: torch.stack(all_classes_preds[task]) for task in all_classes_preds.keys()}
    classes_probabilities = {
        task: torch.softmax(classes_preds_tensor[task], dim=1) for task in classes_preds_tensor.keys()
    }

    assert len(all_reg_preds) == len(texts), "Mismatch in number of predictions and texts"
    assert all(classes_probabilities[task].shape[0] == len(texts) for task in classes_preds), "Mismatch in number of class predictions and texts"

    for i, example in enumerate(examples):
        example["score"] = all_reg_preds[i]
        example["int_score"] = int(all_reg_preds[i])
        for task in classes_preds_tensor.keys():
            probs = classes_probabilities[task][i]
            assert probs.shape[0] == len(SECONDARY_TASKS[task]), f"Mismatch in number of class probabilities for task {task}: expected {len(SECONDARY_TASKS[task])}, got {probs.shape[0]}"
            assert torch.abs(torch.sum(probs) - 1) < 1e-5, f"Probabilities for task {task} do not sum to 1, got {torch.sum(probs)}"
            assert torch.all(probs >= 0), f"Negative probabilities found for task {task}, got {probs[probs < 0]}"

            topk = torch.topk(probs, k=3)
            top_indices = topk.indices.tolist()
            top_probs = topk.values.tolist()

            top_labels = [SECONDARY_TASKS_REVERSED[task][idx] for idx in top_indices]
            for j, (label, prob) in enumerate(zip(top_labels, top_probs)):
                example[f"{task}_class_{j + 1}"] = label
                example[f"{task}_prob_{j + 1}"] = prob

    ShardManager.save_shard_results(shard_number, examples)


def inference_worker(rank: int, shard_number: int, model_path: str):
    print(f"Worker {rank} started for shard {shard_number}", flush=True)
    device = torch.device(f"cuda:{rank}")
    state_dict = torch.load(model_path, map_location=device)
    model = BertMultiTask(MODEL_NAME, [256])
    model.load_state_dict(state_dict)
    model.to(device)

    while True:
        if not ShardManager.check_shard_exists(shard_number):
            print(f"Shard {shard_number} does not exist, exiting.", flush=True)
            break

        if ShardManager.check_shard_processed(shard_number):
            print(f"Shard {shard_number} already processed, skipping.", flush=True)
            shard_number += 4
            continue

        print(f"Processing shard {shard_number}", flush=True)
        infer_shard(model, shard_number, device)
        print(f"Finished processing shard {shard_number}", flush=True)

    print(f"Worker {rank} finished for shard {shard_number}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard_number",
        type=int,
        required=True,
        help="Number of the shard to start with. It will follow the following pattern (assume shard_number is 4k + x): 4k + x, 4k + x + 1, 4k + x + 4, 4k + x + 5, ..., where x is the shard_number passed as an argument.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to be used for inference.",
    )

    args = parser.parse_args()
    shard_number = args.shard_number
    model_path = args.model_path

    processes = []
    world_size = torch.cuda.device_count()
    for rank in range(world_size):
        p = mp.Process(
            target=inference_worker,
            args=(rank, shard_number + rank, model_path),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

