import ijson
import json
from argparse import ArgumentParser
from multiprocessing import Pool
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL_PATH  = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

def get_score_token_count(example):
    tokens = tokenizer(example['text'], return_tensors='pt')
    return tokens.input_ids.shape[1]


def main(in_file, out_file):
    try:
        stats = []
        with open(in_file, 'rb') as f:
            texts = ijson.items(f, 'item')
            with Pool(processes=32) as pool, tqdm(desc=f"Computing token counts for {in_file}", unit="doc") as pbar:
                for count in pool.imap_unordered(get_score_token_count, texts, chunksize=64):
                    stats.append(count)
                    pbar.update()

        with open(out_file, 'w') as out:
            json.dump(stats, out, indent=4, ensure_ascii=False)
        print(f"Saved stats to {out_file}")
    except FileNotFoundError:
        print(f"Error: The file {args.json_file} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {args.json_file} is not a valid JSON file.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Process token statistics from a JSON file.")
    parser.add_argument("--in_file", type=str, help="Path to the JSON file containing token statistics.")
    parser.add_argument("--out_file", type=str, help="Path to save the processed token statistics.")
    args = parser.parse_args()

    main(args.in_file, args.out_file)
