import json
from tqdm import tqdm

TEST_SHARDS = [x for x in range(1, 17)]

ref_examples, out_examples = [], []
for shard_number in tqdm(TEST_SHARDS):
    with open(f"/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results/fineweb2_{shard_number}_results.json", "r") as f:
        ref_examples.extend(json.load(f))

    with open(f"/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_small_results/fineweb2_{shard_number}_results.json", "r") as f:
        out_examples.extend(json.load(f))
print('Loaded shards!', flush=True)

# with open("/export/home/acs/stud/v/vlad_andrei.negoita/app/project/small_model_inference/test_split.json", "w") as f:
#     json.dump({'ref': ref_examples, 'out': out_examples}, f, indent=4, ensure_ascii=False)

assert len(ref_examples) == len(out_examples), "Mismatch in number of examples between reference and output"
topics_matched, formats_matched, age_groups_matched, rmse, mae = 0, 0, 0, 0, 0

for ref, out in tqdm(zip(ref_examples, out_examples), total=len(ref_examples)):
    assert ref["id"] == out["id"], f"ID mismatch: {ref['id']} != {out['id']}"
    
    if ref["topic"] == out["topic_class_1"]:
        topics_matched += 1
    if ref["format"] == out["format_class_1"]:
        formats_matched += 1
    if ref["age_group"] == out["age_group_class_1"]:
        age_groups_matched += 1

    try:
        rmse += (float(ref["int_score"]) - float(out["score"])) ** 2
        mae += abs(float(ref["int_score"]) - float(out["score"]))
    except ValueError as e:
        rmse += 5 ** 2
        mae += 5
        print('Error!', flush=True)


rmse = (rmse / len(ref_examples)) ** 0.5
mae = mae / len(ref_examples)
    
print(f"Topics matched: {topics_matched}/{len(ref_examples)} ({topics_matched / len(ref_examples) * 100:.2f}%)")
print(f"Formats matched: {formats_matched}/{len(ref_examples)} ({formats_matched / len(ref_examples) * 100:.2f}%)")
print(f"Age groups matched: {age_groups_matched}/{len(ref_examples)} ({age_groups_matched / len(ref_examples) * 100:.2f}%)")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
