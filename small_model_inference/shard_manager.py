import os
import json

class ShardManager:
    home_dir = "/export/home/acs/stud/v/vlad_andrei.negoita/"
    data_dir = "fineweb2_ro"
    inference_dir = "fineweb2_ro_small_results"
    inference_dir_backup = "fineweb2_ro_small_backup_results"

    @staticmethod
    def check_shard_exists(shard_number: int) -> bool:
        shard_path = os.path.join(
            ShardManager.home_dir,
            ShardManager.data_dir,
            f"fineweb2_{shard_number}.json",
        )
        return os.path.exists(shard_path)

    @staticmethod
    def check_shard_processed(shard_number: int) -> bool:
        shard_results_path = os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir,
            f"fineweb2_{shard_number}_results.json",
        )
        return os.path.exists(shard_results_path)

    @staticmethod
    def get_shard_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.data_dir,
            f"fineweb2_{shard_number}.json",
        )

    @staticmethod
    def get_shard_results_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir,
            f"fineweb2_{shard_number}_results.json",
        )

    @staticmethod
    def get_shard_results_backup_path(shard_number: int) -> str:
        return os.path.join(
            ShardManager.home_dir,
            ShardManager.inference_dir_backup,
            f"fineweb2_{shard_number}_results_backup.json",
        )

    @staticmethod
    def load_shard(shard_number: int):
        shard_path = ShardManager.get_shard_path(shard_number)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard {shard_number} does not exist at {shard_path}")
        with open(shard_path, 'r') as f:
            return json.load(f)
        
    @staticmethod
    def save_shard_results(shard_number: int, results: list):
        results_path = ShardManager.get_shard_results_path(shard_number)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        backup_results_path = ShardManager.get_shard_results_backup_path(shard_number)
        with open(backup_results_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
