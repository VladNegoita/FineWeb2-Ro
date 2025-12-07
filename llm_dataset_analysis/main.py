import json
import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from training.shard_loader import ShardLoader

def filter_relevant_info_example(example: dict) -> dict:
    return {k: example[k] for k in example if k in ["int_score", "age_group", "topic", "format"]}

def filter_relevant_info_examples(examples: list) -> list:
    return [filter_relevant_info_example(example) for example in tqdm(examples, desc="Filtering examples")]

shard_loader = ShardLoader("/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results")

val_examples = shard_loader.get_examples(shard_loader.get_val_indices())
print(f"Number of validation examples: {len(val_examples)}")

test_examples = shard_loader.get_examples(shard_loader.get_test_indices())
print(f"Number of test examples: {len(test_examples)}")

train_examples = shard_loader.get_examples(shard_loader.get_train_indices())
print(f"Number of training examples: {len(train_examples)}")

filtered_train_examples = filter_relevant_info_examples(train_examples)
filtered_val_examples = filter_relevant_info_examples(val_examples)
filtered_test_examples = filter_relevant_info_examples(test_examples)

filtered_examples = {"train": filtered_train_examples, "val": filtered_val_examples, "test": filtered_test_examples}
with open("filtered_examples.json", "w") as f:
    json.dump(filtered_examples, f, indent=4, ensure_ascii=False)
