from datasets import load_dataset
from multiprocessing import Pool
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL_PATH  = 'meta-llama/Llama-2-7b-hf'
SAMPLE_SIZE = (10 ** 6)
NUM_PROC = 128


fineweb2 = load_dataset(
    "HuggingFaceFW/fineweb-2", split="train", name="ron_Latn", streaming=True
)

print(f"Loading {SAMPLE_SIZE} examples from fineweb-2 dataset...", flush=True)
examples = list(fineweb2.take(SAMPLE_SIZE))
print(f"Loaded {SAMPLE_SIZE} examples from fineweb-2 dataset...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
def get_token_count(example):
    tokens = tokenizer(example['text'], return_tensors='pt')
    return tokens.input_ids.shape[1]

with Pool(NUM_PROC) as pool:
    token_counts = list(
        tqdm(pool.imap_unordered(get_token_count, examples), total=SAMPLE_SIZE, desc="Tokenizing fineweb-2 examples", unit="example")
    )

for max_tokens in [4096]:
    print(f"Max tokens: {max_tokens}", flush=True)
    total_tokens = sum(min(count, max_tokens) for count in token_counts) 
    print(f"Token counts after limiting to {max_tokens} tokens: {total_tokens}...", flush=True)
    print(f"Average tokens per example: {total_tokens / SAMPLE_SIZE:.2f}", flush=True)
    print(f"Median tokens per example: {sorted(token_counts)[SAMPLE_SIZE // 2]}", flush=True)

print("Finished tokenizing examples", flush=True)