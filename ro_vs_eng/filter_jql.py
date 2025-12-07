import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

GEMMA = 'Edu-JQL-Gemma-SF'
MISTRAL = 'Edu-JQL-Mistral-SF'
LLAMA = 'Edu-JQL-Llama-SF'

LLMS = [GEMMA, MISTRAL, LLAMA]

def process_file(args):
    filename, input_dir, output_dir, min_scores = args
    if not filename.endswith('.json'):
        return f"Skipped non-JSON file: {filename}"

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading {filename}: {e}"

    filtered = [item for item in data if all(item[f'score_{llm}'] >= min_scores[llm] for llm in LLMS)]

    try:
        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return f"Error writing {filename}: {e}"

    return f"Processed {filename}: {len(filtered)} items"


def main(input_dir: str, output_dir: str, min_score_values: dict):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")
    print(f"Input directory exists: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
    
    files = os.listdir(input_dir)
    tasks = [(fn, input_dir, output_dir, min_score_values) for fn in files]

    print(f"Starting processing with {cpu_count()} workers and {min_score_values} thresholds...")
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            print(result)

    print("All files processed.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Filter JSON files in parallel based on score.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input JSON directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for filtered output files.")

    parser.add_argument("--min_score_gemma", type=float, default=3.46484375, help="Minimum score value for filtering Gemma.")
    parser.add_argument("--min_score_mistral", type=float, default=2.439453125, help="Minimum score value for filtering Mistral.")
    parser.add_argument("--min_score_llama", type=float, default=2.8125, help="Minimum score value for filtering Llama.")

    args = parser.parse_args()
    min_score = {
        GEMMA: args.min_score_gemma,
        MISTRAL: args.min_score_mistral,
        LLAMA: args.min_score_llama
    }

    main(args.input_dir, args.output_dir, min_score)
