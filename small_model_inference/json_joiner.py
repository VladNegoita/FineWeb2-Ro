import json
from argparse import ArgumentParser
import os
from tqdm import tqdm

def main(input_dir: str, output_file: str):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")
    print(f"Input directory exists: {input_dir}")

    all_data = []
    print(f"Started joining files in {input_dir}")
    
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            print('Skipping non-JSON file:', filename)
            continue

        input_file_path = os.path.join(input_dir, filename)
        with open(input_file_path, 'r') as file:
            data = json.load(file)
            all_data.extend(data)

    print(f"Total items collected: {len(all_data)}")
    with open(output_file, 'w') as file:
        json.dump(all_data, file, indent=4, ensure_ascii=False)

    print(f"Joined data saved to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Join multiple JSON files into a single JSON file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input JSON directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    main(args.input_dir, args.output_file)