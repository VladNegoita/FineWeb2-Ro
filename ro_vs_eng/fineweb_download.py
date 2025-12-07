from datasets import load_dataset
import json
import os
from tqdm import tqdm

DIR_PATH = "/export/home/acs/stud/v/vlad_andrei.negoita/"
DATASET = "fineweb"

DATASET_PATH = os.path.join(DIR_PATH, DATASET)
SHARD_SIZE = 1300
TOTAL_EXAMPLES = 54128784

FIELDS = [
    "id",
    "text",
    "url",
    "date",
    "dump",
    "file_path",
    "language_score",
]


def write_examples(examples, file_idx):
    with open(
        os.path.join(DIR_PATH, DATASET, f"fineweb_{file_idx}.json"),
        "w",
    ) as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)


def main():
    os.makedirs(DATASET_PATH, exist_ok=True)

    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", streaming=True, split="train", name="sample-100BT"
    )

    examples = []
    for idx, example in tqdm(enumerate(fineweb), total=TOTAL_EXAMPLES):
        example_dict = {field: example[field] for field in FIELDS}
        examples.append(example_dict)
        if len(examples) == SHARD_SIZE:
            write_examples(examples, idx // SHARD_SIZE)
            examples = []

        if idx >= TOTAL_EXAMPLES:
            break

    if len(examples) > 0:
        write_examples(examples, idx // SHARD_SIZE + 1)
        examples = []


if __name__ == "__main__":
    main()
