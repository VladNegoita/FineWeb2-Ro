import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import reduce
import numpy as np


GEMMA = 'Edu-JQL-Gemma-SF'
MISTRAL = 'Edu-JQL-Mistral-SF'
LLAMA = 'Edu-JQL-Llama-SF'

LLMS = [GEMMA, MISTRAL, LLAMA]

def merge_counters(counter_list):
    def merge_acc(acc, current):
        for key in LLMS:
            acc[key] += current[key]
        return acc

    return reduce(merge_acc, counter_list, {key: [] for key in LLMS})


def process_file(file_path):
    counters = {key: [] for key in LLMS}
    try:
        with open(file_path, "r") as file:
            examples = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return counters
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return counters

    for example in examples:
        for llm in LLMS:
            counters[llm].append(example[f'score_{llm}'])

    return counters

def main(dataset, base_path):
    input_dir = os.path.join(base_path, dataset)
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    files.sort()
    print(f"Found {len(files)} files in the input directory.")
    processes_count = min(cpu_count(), 32, len(files))

    with Pool(processes=processes_count) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc=f"Processing files using {processes_count} processes"))

    counters = merge_counters(results)
    print("Analysis complete. Summary:")

    percentiles = {}

    for llm in LLMS:
        values = np.array(counters[llm])
        percentiles[llm] = {
            p: float(np.percentile(values, p)) for p in [90 + 0.5 * x for x in range(1, 11)]
        }
        print(f"{llm} percentiles:")
        for k, v in percentiles[llm].items():
            print(f"  {k}: {v}")

        print("\n", flush=True)

    with open(f"ro_vs_eng/llm_percentiles90.json", "w") as summary_file:
        json.dump(percentiles, summary_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Full dataset analysis script")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        required=True,
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Path to the base directory",
        required=True,
    )

    args = parser.parse_args()
    main(args.dataset, args.base_path)