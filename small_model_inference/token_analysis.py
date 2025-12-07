import json
from argparse import ArgumentParser
import numpy as np

def main(in_file):
    with open(in_file, 'r') as f:
        data = json.load(f)

    if not data:
        print("No data found in the file.", flush=True)
        return

    print(f"Loaded {len(data)} records from {in_file}", flush=True)
    data = np.array([token_count for token_count in data])
    print(f"Converted data to numpy array", flush=True)
    for threshold in [4096, 2 ** 31]:
        print(f"Threshold: {threshold}")
        truncated_data = np.minimum(data, threshold)
        total_tokens = np.sum(truncated_data)
        print(f"Total token count: {total_tokens / 1e9:.2f}B")
        print("-" * 40, end="\n\n", flush=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Process token statistics from a JSON file.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the JSON file containing token statistics.")
    args = parser.parse_args()
    main(args.in_file)