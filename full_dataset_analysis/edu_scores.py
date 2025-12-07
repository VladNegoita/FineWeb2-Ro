import torch
import json
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm.auto import tqdm

DATASET_NAME = "HuggingFaceFW/fineweb"
CLASSIFIER_MODEL = "HuggingFaceFW/fineweb-edu-classifier"
OUTPUT_FILE = "full_dataset_analysis/edu.json"
MAX_EXAMPLES = 1_000_000
BATCH_SIZE = 64
DEVICE = torch.device("cuda")

ds = load_dataset(DATASET_NAME, split="train", streaming=True)
ds = ds.take(MAX_EXAMPLES)

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL)
model.to(DEVICE)
model.eval()

def score_batch(batch):
    enc = tokenizer(batch['text'], padding=True, truncation=True,
                     max_length=tokenizer.model_max_length, return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    
    scores = out.logits.squeeze(-1).tolist() # regression output
    return {'scores': scores}

scored_iter = ds.map(
    score_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=[c for c in ds.column_names if c != 'text'],
)

scores = []
for idx, example in enumerate(tqdm(scored_iter, total=math.ceil(MAX_EXAMPLES), desc="Processing batches")):
    scores.append(round(float(example['scores']), 2))

with open(OUTPUT_FILE, 'w') as f:
    json.dump(scores, f)

print(f"Scores saved to {OUTPUT_FILE}, total processed: {len(scores)}")
