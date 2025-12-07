import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import reduce


def merge_counters(counter_list):
    def merge_acc(acc, current):
        for key in ['topic', 'format']:
            for k, v in current[key].items():
                if k not in acc[key]:
                    acc[key][k] = 0
                acc[key][k] += v
        return acc

    return reduce(merge_acc, counter_list, {'topic': dict(), 'format': dict()})


def process_file(file_path_threshold):
    file_path, threshold = file_path_threshold
    counters = {'topic': dict(), 'format': dict()}
    try:
        with open(file_path, "r") as file:
            examples = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return counters
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return counters
    

    for example in filter(lambda x: x.get('score', 0) >= threshold, examples):
        for secondary_task in ['topic', 'format']:
            value = example[f"{secondary_task}_class_1"]
            if value not in counters[secondary_task]:
                counters[secondary_task][value] = 0
            counters[secondary_task][value] += 1

    return counters

def main(dataset, base_dir, threshold):
    input_dir = os.path.join(base_dir, dataset)
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    files.sort()
    print(f"Found {len(files)} files in the input directory.")
    processes_count = min(cpu_count(), 32, len(files))

    with Pool(processes=processes_count) as pool:
        results = list(tqdm(pool.imap(process_file, zip(files, (threshold for _ in range(len(files))))), total=len(files), desc=f"Processing files using {processes_count} processes"))

    counters = merge_counters(results)
    print("Analysis complete. Summary:")
    print(f"Total examples processed: {sum(counters['topic'].values())}")

    print("\nTopic distribution:")
    for topic, count in counters['topic'].items():
        print(f"  {topic}: {count}")

    print("\nFormat distribution:")
    for format_type, count in counters['format'].items():
        print(f"  {format_type}: {count}")

    with open(f"ro_vs_eng/{dataset}_{threshold}.json", "w") as summary_file:
        json.dump(counters, summary_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Full dataset analysis script")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        required=True,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Path to the base directory",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for filtering examples",
        required=False,
        default=0.0
    )

    args = parser.parse_args()
    main(args.dataset, args.base_dir, args.threshold)