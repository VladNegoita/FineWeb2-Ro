import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import reduce

ADULT_CONTENT = 'Conținut pentru adulți'

def process_file(file_path):
    try:
        with open(file_path, "r") as file:
            examples = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

    return [example for example in examples if example['topic_class_1'] == ADULT_CONTENT and example['score'] >= 3]



def merge_counters(counter_list):
    def merge_acc(acc, current):
        acc.extend(current)
        return acc
    
    return reduce(merge_acc, counter_list, [])


def main(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    files.sort()
    print(f"Found {len(files)} files in the input directory.")

    processes_count = min(cpu_count(), 32, len(files))
    with Pool(processes=processes_count) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc=f"Processing files using {processes_count} processes"))

    counters = merge_counters(results)
    print("Analysis complete!")

    with open("full_dataset_analysis/adult3.json", "w") as summary_file:
        json.dump(counters, summary_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Full dataset analysis script")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the sharded dataset (directory containing JSON files)",
        required=True,
    )
    args = parser.parse_args()
    main(args.input_dir)