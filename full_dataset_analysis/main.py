import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import reduce

def process_file(file_path):
    counters = {'score': [], 'topic': dict(), 'age_group': dict(), 'format': dict()}
    try:
        with open(file_path, "r") as file:
            examples = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
    
    for example in examples:
        counters['score'].append(round(example['score'], 2))
        for secondary_task in ['topic', 'format', 'age_group']:
            value = example[f"{secondary_task}_class_1"]
            if value not in counters[secondary_task]:
                counters[secondary_task][value] = 0
            counters[secondary_task][value] += 1

    return counters


def merge_counters(counter_list):
    def merge_acc(acc, current):
        acc['score'].extend(current['score'])
        for key in ['topic', 'age_group', 'format']:
            for k, v in current[key].items():
                if k not in acc[key]:
                    acc[key][k] = 0
                acc[key][k] += v
        return acc
    
    return reduce(merge_acc, counter_list, {'score': [], 'topic': dict(), 'age_group': dict(), 'format': dict()})

def main(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    files.sort()
    print(f"Found {len(files)} files in the input directory.")
    processes_count = min(cpu_count(), 32, len(files))
    with Pool(processes=processes_count) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc=f"Processing files using {processes_count} processes"))

    counters = merge_counters(results)
    print("Analysis complete. Summary:")
    print(f"Total examples processed: {len(counters['score'])}")
    print(f"Score range: {min(counters['score'])} to {max(counters['score'])}")

    print("\nTopic distribution:")
    for topic, count in counters['topic'].items():
        print(f"  {topic}: {count}")

    print("\nFormat distribution:")
    for format_type, count in counters['format'].items():
        print(f"  {format_type}: {count}")

    print("\nAge group distribution:")
    for age_group, count in counters['age_group'].items():
        print(f"  {age_group}: {count}")

    with open("full_dataset_analysis/summary.json", "w") as summary_file:
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