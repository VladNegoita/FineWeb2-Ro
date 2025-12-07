import os
import gc
import sys
import json
import math
import time
import torch
import argparse
import torch.multiprocessing as mp
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch._dynamo.config.cache_size_limit = 64

from prompts.final_ro import all_prompt_ro_gemma_without_text

FIELDS = {
    "Topic": "topic",
    "Subtopic": "subtopic",
    "Format": "format",
    "Nivel educațional": "age_group",
    "Valoare educațională": "int_score",
    "Explicație": "explanation",
}
model_id = "google/gemma-3-12b-it"
context_length = 32768
output_length = 512


class ShardManager:
    home_dir = "/export/home/acs/stud/v/vlad_andrei.negoita/"
    data_dir = "fineweb2_ro"
    inference_dir = "fineweb2_ro_results"
    inference_dir_backup = "fineweb2_ro_backup_results"

    @staticmethod
    def check_shard_exists(shard_number: int) -> bool:
        shard_path = os.path.join(
            ShardManager.home_dir,
            ShardManager.data_dir,
            f"fineweb2_{shard_number}.json",
        )
        return os.path.exists(shard_path)

    @staticmethod
    def check_shard_processed(shard_number: int) -> bool:
        shard_results_path = os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir,
            f"fineweb2_{shard_number}_results.json",
        )
        return os.path.exists(shard_results_path)

    @staticmethod
    def get_shard_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.data_dir,
            f"fineweb2_{shard_number}.json",
        )

    @staticmethod
    def get_shard_results_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir,
            f"fineweb2_{shard_number}_results.json",
        )

    @staticmethod
    def get_shard_results_backup_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir_backup,
            f"fineweb2_{shard_number}_results_backup.json",
        )


def extract_prediction(output: str, result_dict: dict):
    problem_output = False
    for line in [x.strip() for x in output.split("\n")]:
        while len(line) > 0 and line.startswith("*"):
            line = line[1:].strip()

        if len(line) == 0 or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while len(key) > 0 and key[0] == "*":
            key = key[1:].strip()
        while len(value) > 0 and value[0] == "*":
            value = value[1:].strip()

        if key in FIELDS.keys():
            if FIELDS[key] == "int_score" and value.isdigit():
                result_dict[FIELDS[key]] = int(value)
            else:
                result_dict[FIELDS[key]] = value
        else:
            problem_output = True

    for value in FIELDS.values():
        if value not in result_dict:
            result_dict[value] = "ERROR"
            problem_output = True

    if problem_output:
        print(f"Problem output!")


def load_examples(file_path: str):
    with open(file_path) as f:
        examples = json.load(f)
    return examples


def save_results(results, shard_number: int):
    with open(ShardManager.get_shard_results_path(shard_number), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(ShardManager.get_shard_results_backup_path(shard_number), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def get_prompt_messages():
    return all_prompt_ro_gemma_without_text()


def split_prompt(prompt: str):
    before, _, after = prompt.partition("%s")
    return before, after


def clean_cuda_memory(device=None):
    gc.collect()
    if device:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    gc.collect()


def inference_worker(rank, examples, return_dict):
    print(f"Rank {rank} starting worker", flush=True)

    device = torch.device(f"cuda:{rank}")

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model = torch.compile(model, dynamic=True)
    model.eval()

    print(f"Rank {rank} loaded model", flush=True)

    # Common prompt for all examples
    messages = get_prompt_messages()
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    before, after = split_prompt(prompt)
    before_ids = processor.tokenizer(
        before, return_tensors="pt", add_special_tokens=False
    ).input_ids
    after_ids = processor.tokenizer(
        after, return_tensors="pt", add_special_tokens=False
    ).input_ids

    outputs = []
    for idx, example in enumerate(examples):
        gc.collect()

        start_time = time.time()
        text_ids = processor.tokenizer(
            example["text"],
            return_tensors="pt",
            add_special_tokens=False,
            max_length=context_length
            - output_length
            - before_ids.shape[-1]
            - after_ids.shape[-1],
            truncation=True,
        ).input_ids
        full_ids = torch.cat([before_ids[0], text_ids[0], after_ids[0]], dim=0)
        batch_encoding = processor.tokenizer.pad(
            {"input_ids": [full_ids.tolist()]},
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids = batch_encoding["input_ids"].to(device)
        attention_mask = batch_encoding["attention_mask"].to(device)
        input_len = input_ids.shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_length,
                do_sample=False,
                top_k=None,
                top_p=None,
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        labeled_example = example.copy()
        labeled_example["output"] = decoded
        extract_prediction(decoded, labeled_example)
        outputs.append(labeled_example)

        elapsed_time = time.time() - start_time
        print(
            f"Rank {rank} finished example {idx + 1}/{len(examples)} in {elapsed_time:.3f} seconds",
            flush=True,
        )
        _ = input_ids.detach().cpu()
        _ = attention_mask.detach().cpu()
        _ = generation.detach().cpu()
        del text_ids, full_ids, batch_encoding, input_ids, attention_mask, generation

    return_dict[rank] = outputs
    print(f"Rank {rank} finished processing examples", flush=True)

    del model, processor, before_ids, after_ids
    clean_cuda_memory(device)


def infer_shard(shard_number: int):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    world_size = 2
    examples = load_examples(ShardManager.get_shard_path(shard_number))

    split_size = math.ceil(len(examples) / world_size)
    splits = [
        examples[i * split_size : (i + 1) * split_size] for i in range(world_size)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(world_size):
        p = mp.Process(
            target=inference_worker,
            args=(rank, splits[rank], return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    for rank in range(world_size):
        results.extend(return_dict[rank])

    save_results(results, shard_number)
    print("All processes finished, results saved!", flush=True)
    clean_cuda_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard_number",
        type=int,
        required=True,
        help="Number of the shard to start with.",
    )
    args = parser.parse_args()
    shard_number = args.shard_number

    print(f"Started looking for {shard_number}", flush=True)
    while True:
        if not ShardManager.check_shard_exists(shard_number):
            print(f"Shard {shard_number} does not exist, exiting.", flush=True)
            break

        if ShardManager.check_shard_processed(shard_number):
            print(f"Shard {shard_number} already processed, skipping.", flush=True)
            shard_number += 2
            continue

        print(f"Processing shard {shard_number}", flush=True)
        infer_shard(shard_number)
        print(f"Finished processing shard {shard_number}", flush=True)

        shard_number += 2


if __name__ == "__main__":
    main()
