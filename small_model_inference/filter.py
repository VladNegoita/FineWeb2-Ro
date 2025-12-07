import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_file(args):
    filename, input_dir, output_dir, min_score = args
    if not filename.endswith('.json'):
        return f"Skipped non-JSON file: {filename}"

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading {filename}: {e}"

    filtered = [item for item in data if item['score'] >= min_score]

    try:
        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return f"Error writing {filename}: {e}"

    return f"Processed {filename}: {len(filtered)} items"


def main(input_dir: str, output_dir: str, min_score_value: float):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")
    print(f"Input directory exists: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
    
    # Prepare arguments for each file
    files = os.listdir(input_dir)
    tasks = [(fn, input_dir, output_dir, min_score_value) for fn in files]

    print(f"Starting processing with {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        # Use imap_unordered for progress
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            print(result)

    print("All files processed.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Filter JSON files in parallel based on score.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input JSON directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for filtered output files.")
    parser.add_argument("--min_score_value", type=float, required=True, help="Minimum score threshold.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.min_score_value)
