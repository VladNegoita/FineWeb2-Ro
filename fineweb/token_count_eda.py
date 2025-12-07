from datasets import load_dataset
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.llama3 import llama3_70b
from prompts.final_eng import all_prompt_eng
from prompts.final_ro import all_prompt_ro


def get_token_count_overhead(prompt_fun) -> int:
    tokenizer = AutoTokenizer.from_pretrained(llama3_70b[0])
    input_ids = tokenizer.apply_chat_template(
        prompt_fun(""), add_generation_prompt=True, return_tensors="pt"
    )
    return input_ids.size(1)


fineweb_edu = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
fineweb2 = load_dataset(
    "HuggingFaceFW/fineweb-2", split="train", name="ron_Latn", streaming=True
)

sample_size = 2 * (10**5)

fineweb_edu = fineweb_edu.take(sample_size)
fineweb2 = fineweb2.take(sample_size)

token_count_fineweb_edu = [
    min(x["token_count"], 8192) for x in tqdm(fineweb_edu, total=sample_size)
]

print("Finished loading datasets", flush=True)

fineweb2_texts = [x["text"] for x in fineweb2]
tokenizer = AutoTokenizer.from_pretrained(llama3_70b[0])
with Pool(64) as pool:
    tokenized_fineweb2 = list(
        tqdm(pool.imap_unordered(tokenizer, fineweb2_texts), total=sample_size)
    )
token_count_fineweb2 = [min(len(x["input_ids"]), 8192) for x in tokenized_fineweb2]

print("Finished tokenizing examples", flush=True)

for dataset, token_count in [
    ("fineweb-edu", token_count_fineweb_edu),
    ("fineweb-2", token_count_fineweb2),
]:
    sns.histplot(token_count, kde=True, label=dataset)
    plt.title(f"Token count distribution for {dataset}")
    plt.xlabel("Token count")
    plt.ylabel("Density")
    plt.savefig(f"photos/token_count_eda_{dataset}.png")
    plt.clf()

with open("fineweb/stats.txt", "wt") as f:
    f.write(
        f"Full prompt in eng overhead: {get_token_count_overhead(all_prompt_eng)} tokens\n"
    )
    f.write(
        f"Full prompt in ro overhead: {get_token_count_overhead(all_prompt_ro)} tokens\n"
    )
    f.write(
        f"Percentage of examples with token count >= 8192 in fineweb-edu: {sum(x >= 8192 for x in token_count_fineweb_edu) / len(token_count_fineweb_edu) * 100:.2f}%\n"
    )
    f.write(
        f"Percentage of examples with token count >= 8192 in fineweb-2 (romanian): {sum(x >= 8192 for x in token_count_fineweb2) / len(token_count_fineweb2) * 100:.2f}%\n"
    )
