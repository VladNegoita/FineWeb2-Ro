from datasets import load_dataset
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.llama3 import infer_llama3, llama3_70b
from prompts.translator import translate_prompt

ABSENT_FEATURE = "N/A"  # to be manually labeled later

loaded_model = infer_llama3(
    llama3_70b, max_output_tokens=llama3_70b[1] / 2
)  # we assume that the translation will be rougly equal in length to the original text

# We remove id (since the data is coming from multiple sources) and token count (it's not invariant to translation)
# We also remove dataset specific information (language scores, minhashes, etc.)
examples = []
fineweb_edu = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
for i, example in enumerate(fineweb_edu):
    if len(examples) == 50:
        break

    try:
        translated_text = translate_prompt(example["text"])
    except Exception as e:
        print(f"Failed to translate example {i} from fineweb-edu due to error: {e}")
        continue

    examples.append(
        {
            "original_text": example["text"],
            "text": loaded_model(translate_prompt(example["text"])),
            "dump": example["dump"],
            "url": example["url"],
            "date": example["date"],
            "file_path": example["file_path"],
            "score": example[
                "score"
            ],  # not sure if we need this or not, keep it just in case
            "int_score": example["int_score"],
            "topic": ABSENT_FEATURE,
            "subtopic": ABSENT_FEATURE,
            "format": ABSENT_FEATURE,
            "age_group": ABSENT_FEATURE,
            "dataset_source": "fineweb-edu",
        }
    )
    print(f"Processed example {i} from fineweb-edu")

fineweb2 = load_dataset(
    "HuggingFaceFW/fineweb-2", split="train", name="ron_Latn", streaming=True
)
for i, example in enumerate(fineweb2):
    if len(examples) == 100:
        break
    examples.append(
        {
            "text": example["text"],
            "dump": example["dump"],
            "url": example["url"],
            "date": example["date"],
            "file_path": example["file_path"],
            "int_score": ABSENT_FEATURE,
            "topic": ABSENT_FEATURE,
            "subtopic": ABSENT_FEATURE,
            "format": ABSENT_FEATURE,
            "age_group": ABSENT_FEATURE,
            "dataset_source": "fineweb-2",
        }
    )
    print(f"Processed example {i} from fineweb-2")

with open("datasets/small_dataset2.json", "w") as f:
    json.dump(examples, f, indent=4, ensure_ascii=False)
