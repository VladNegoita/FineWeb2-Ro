from argparse import ArgumentParser
import ijson
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute topic and format counts from a JSON file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    counts = {
        "topic": {},
        "format": {},
        "age_group": {},
    }

    with open(args.input_file, 'r') as infile:
        examples = ijson.items(infile, 'item')
        for example in tqdm(examples, total=3883932):
            topic, fmt, age_g = example["topic_class_1"], example["format_class_1"], example["age_group_class_1"]
            
            if topic not in counts["topic"]:
                counts["topic"][topic] = 0
            counts["topic"][topic] += 1
            
            if fmt not in counts["format"]:
                counts["format"][fmt] = 0
            counts["format"][fmt] += 1

            if age_g not in counts["age_group"]:
                counts["age_group"][age_g] = 0
            counts["age_group"][age_g] += 1
        
    with open(args.output_file, "w") as outfile:
        json.dump(counts, outfile, indent=4, ensure_ascii=False)
    
    print(f"Filtered data saved to {args.output_file}.")
