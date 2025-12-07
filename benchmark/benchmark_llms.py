import json
import os
import sys
from tqdm import tqdm
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.llama3 import (
    infer_llama3,
    llama3_8b,
    llama3_1_8b,
    llama3_70b,
    llama3_1_70b,
    llama3_3_70b,
)
from inference.qwen import infer_qwen, qwen_2_5
from inference.cohere import infer_cohere, cohere_35b
from inference.gemma3 import infer_gemma3, gemma3_27b, gemma3_12b
from inference.gemma2 import infer_gemma2, gemma2_27b, gemma2_9b
from inference.mistral import infer_mistral, mistral_24b
from prompts.final_eng import all_prompt_eng, all_prompt_eng_gemma, all_prompt_eng_user
from prompts.final_ro import all_prompt_ro, all_prompt_ro_gemma, all_prompt_ro_user

MAX_NEW_TOKENS = 512  # we expect the output to have at most ~256 words
FIELDS = {
    "Topic": "topic",
    "Subtopic": "subtopic",
    "Format": "format",
    "Nivel educațional": "age_group",
    "Valoare educațională": "int_score",
    "Explicație": "explanation",
}
COMPARABLE_FIELDS = {"topic", "format", "age_group", "int_score"}


def extract_prediction(output: str):
    d = {v: "ERROR" for k, v in FIELDS.items()}
    for line in [x.strip() for x in output.split("\n")]:
        while len(line) > 0 and line.startswith("*"):
            line = line[1:].strip()

        if len(line) == 0 or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while len(key) > 0 and key[0] == "*":
            key = key[1:].strip()
        while len(value) > 0 and value[0] == "*":
            value = value[1:].strip()

        if key in FIELDS.keys():
            d[FIELDS[key]] = value
            if FIELDS[key] == "int_score" and value.isdigit():
                d[FIELDS[key]] = int(value)
        else:
            print(f"Extracted garbage: {key} -> {value}")
    return d


def main():
    with open("datasets/small_dataset.json") as f:
        small_dataset = json.load(f)

    for model_spec in [gemma3_12b]:
        model = infer_gemma3(model_spec, max_output_tokens=MAX_NEW_TOKENS)
        score = {k: 0 for k in COMPARABLE_FIELDS}
        predictions = []
        for example in tqdm(small_dataset):
            prompt = all_prompt_ro_gemma(example["text"])
            output = model(prompt)
            prediction = extract_prediction(output)
            prediction["text"] = example["text"]
            prediction["output"] = output
            prediction["prompt"] = prompt

            for field in COMPARABLE_FIELDS:
                prediction[f"original_{field}"] = example[field]
                if prediction[field] == example[field]:
                    score[field] += 1

            predictions.append(prediction)

        print(f"Model {model_spec[0]} finished, scores: {score}")
        with open(
            f"predictions/small_dataset_{model_spec[0].split('/')[-1]}_final.json",
            "w",
        ) as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)

        gc.collect()


if __name__ == "__main__":
    main()
