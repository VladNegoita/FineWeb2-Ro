import os
import json
import sys
from typing import List


class ShardLoader:
    TEST_SHARDS = [x for x in range(1, 17)]
    VAL_SHARDS = [x for x in range(17, 25)]
    # Train shards are the rest

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def get_shard_path(self, index: int) -> str:
        return os.path.join(self.dir_path, f"fineweb2_{index}_results.json")

    def get_test_indices(self) -> List[int]:
        return ShardLoader.TEST_SHARDS

    def get_val_indices(self) -> List[int]:
        return ShardLoader.VAL_SHARDS

    def get_train_indices(self) -> List[int]:
        indices = []
        current_index = 0
        while True:
            if (
                current_index in ShardLoader.TEST_SHARDS
                or current_index in ShardLoader.VAL_SHARDS
            ):
                current_index += 1
                continue

            file_path = self.get_shard_path(current_index)
            if not os.path.exists(file_path):
                break

            indices.append(current_index)
            current_index += 1

        return indices

    def get_examples(self, indices: List[int]) -> List[dict]:
        examples = []
        for index in indices:
            file_path = self.get_shard_path(index)

            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                sys.exit(1)

            with open(file_path, "r") as f:
                examples.extend(json.load(f))
        return examples


if __name__ == "__main__":
    dir_path = "/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results"
    shard_manager = ShardLoader(dir_path)
    test_indices = shard_manager.get_test_indices()
    val_indices = shard_manager.get_val_indices()
    train_indices = shard_manager.get_train_indices()

    print(f"Test indices: {test_indices}")
    print(f"val indices: {val_indices}")
    print(f"Train indices: {train_indices}")

    assert len(test_indices) == 16
    assert len(val_indices) == 8

    assert set(train_indices).intersection(set(test_indices)) == set()
    assert set(train_indices).intersection(set(val_indices)) == set()
    assert set(test_indices).intersection(set(val_indices)) == set()

    assert len(set(train_indices)) == len(train_indices)
    assert len(set(test_indices)) == len(test_indices)
    assert len(set(val_indices)) == len(val_indices)

    test_examples = shard_manager.get_examples(test_indices)
    val_examples = shard_manager.get_examples(val_indices)
    train_examples = shard_manager.get_examples(train_indices)

    print(f"Number of test examples: {len(test_examples)}")
    print(f"Number of val examples: {len(val_examples)}")
    print(f"Number of train examples: {len(train_examples)}")
