from datasets import load_dataset
import json
import os
from tqdm import tqdm

DIR_PATH = "/export/home/acs/stud/v/vlad_andrei.negoita/"
DATASET = "fineweb2_ro"
BACKUP = "fineweb2_ro_backup"

DATASET_PATH = os.path.join(DIR_PATH, DATASET)
BACKUP_PATH = os.path.join(DIR_PATH, BACKUP)
SHARD_SIZE = 1300

FIELDS = [
    "id",
    "text",
    "url",
    "date",
    "dump",
    "file_path",
    "language_score",
    "minhash_cluster_size",
    "top_langs",
]


def write_examples(examples, file_idx):
    with open(
        os.path.join(DIR_PATH, DATASET, f"fineweb2_{file_idx}.json"),
        "w",
    ) as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)
    with open(
        os.path.join(DIR_PATH, BACKUP, f"fineweb2_{file_idx}.json"),
        "w",
    ) as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)


def main():
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(BACKUP_PATH, exist_ok=True)

    fineweb2 = load_dataset(
        "HuggingFaceFW/fineweb-2", split="train", name="ron_Latn", streaming=True
    )

    examples = []
    for idx, example in tqdm(enumerate(fineweb2), total=54128784):
        example_dict = {field: example[field] for field in FIELDS}
        examples.append(example_dict)
        if len(examples) == SHARD_SIZE:
            write_examples(examples, idx // SHARD_SIZE)
            examples = []

    if len(examples) > 0:
        write_examples(examples, idx // SHARD_SIZE + 1)
        examples = []


if __name__ == "__main__":
    main()
