import os
import torch
import json
from torch.amp import autocast
from argparse import ArgumentParser
import torch.multiprocessing as mp
from jql.src.utils.regression_head import RegressionHead
from jql.src.utils.embedder import get_embedder_instance
from transformers.utils.hub import cached_file

BATCH_SIZE = 16

def annotate(base_model, regression_heads, examples, device):
    for start_idx in range(0, len(examples), BATCH_SIZE):
        batch = examples[start_idx:start_idx + BATCH_SIZE]
        with torch.no_grad(), autocast("cuda"):
            embeddings = base_model.embed([ex["text"] for ex in batch])
            outputs = dict()
            for name, regression_head in regression_heads.items():
                outputs[name] = regression_head(embeddings).cpu().squeeze(1)
            for i, _ in enumerate(batch):
                for name in outputs:
                    examples[start_idx + i][f'score_{name}'] = float(outputs[name][i].item())


def worker(rank: int, base_path: str, offset: int):
    print(f"Worker {rank} started...")

    # Load models and tokenizers
    print("Loading models and tokenizers...", flush=True)
    device = torch.device(f"cuda:{rank}")

    base_model = get_embedder_instance('Snowflake/snowflake-arctic-embed-m-v2.0', device, torch.bfloat16)

    regression_head_checkpoints = {
        'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
        'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
        'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads', 'checkpoints/edu-llama-snowflake-balanced.ckpt'),
    }

    regression_heads = {}
    for name, path in regression_head_checkpoints.items():
        regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(torch.bfloat16)

    print("Models loaded.", flush=True)

    in_dir = os.path.join(base_path, "fineweb2_ro")
    out_dir = os.path.join(base_path, "fineweb2_ro_jql")

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(in_dir):
        raise ValueError(f"Input directory {in_dir} does not exist.")

    curr_file_idx = offset + rank
    while True:
        curr_file = os.path.join(in_dir, f"fineweb2_{curr_file_idx}.json")
        if not os.path.exists(curr_file):
            print(f"File {curr_file} does not exist. Stopping.")
            break

        if os.path.exists(os.path.join(out_dir, f"fineweb2_{curr_file_idx}_jql.json")):
            print(f"Annotated file {curr_file} already exists. Skipping.", flush=True)
            curr_file_idx += 4
            continue

        with open(curr_file, "r") as f:
            examples = json.load(f)

        print(f"Processing file {curr_file} with {len(examples)} examples.", flush=True)
        annotate(base_model, regression_heads, examples, device)
        with open(os.path.join(out_dir, f"fineweb2_{curr_file_idx}_jql.json"), "w") as f:
            json.dump(examples, f, indent=4, ensure_ascii=False)

        curr_file_idx += 4
        print(f"Finished processing file {curr_file}.", flush=True)


def main(base_path: str, offset: int):
    print(f"Starting processing for fineweb2...", flush=True)
    mp.spawn(worker, args=(base_path, offset), nprocs=2, join=True)
    print("Processing completed.", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process fineweb2 dataset.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for the fineweb2 sharded dataset.")
    parser.add_argument("--offset", type=int, required=True, help="Offset for the current workers.")
    args = parser.parse_args()
    main(args.base_path, args.offset)