import json
from datasets import Dataset, Features, Value
import os
import glob

# def my_batched_function(batch):
#     batch["int_score"] = [round(score) for score in batch["score"]]
#     return batch

def generate_examples():
    print("Starting generator... processing files.", flush=True)
    for filepath in glob.iglob(file_pattern):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            for example in data:
                example['int_score'] = str(example['int_score'])
                yield example

data_dir = "/export/home/acs/stud/v/vlad_andrei.negoita/fineweb2_ro_results"
file_pattern = os.path.join(data_dir, "*.json")

features = Features({
    'id': Value("string"),
    'text': Value("string"),
    'url': Value("string"),
    'date': Value("string"),
    'dump': Value("string"),
    'file_path': Value("string"),
    'language_score': Value("float64"),
    'minhash_cluster_size': Value("int64"),
    'top_langs': Value("string"),
    'output': Value("string"),
    'explanation': Value("string"),
    'int_score': Value("string"), # because of errors
    'age_group': Value("string"),
    'topic': Value("string"),
    'subtopic': Value("string"),
    'format': Value("string")
})

print("Loading dataset...", flush=True)
dataset_stream = Dataset.from_generator(generate_examples, features=features)
# dataset_stream = load_dataset(
#     "json",
#     data_files=file_pattern,
#     split="train",
#     features=features,
# )

# print("Applying .map() to the stream...", flush=True)
# mapped_stream = dataset_stream.map(my_batched_function, batched=True, batch_size=1300)


org_repo_name = "OpenLLM-Ro/fineweb2-ro-llm-annotated"
print(f"Pushing processed stream to {org_repo_name}...", flush=True)

dataset_stream.push_to_hub(
    org_repo_name,
    max_shard_size="500MB"
)

print("Done! Processed dataset pushed successfully.", flush=True)