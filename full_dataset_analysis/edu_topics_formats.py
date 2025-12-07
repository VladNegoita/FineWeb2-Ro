import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import json
from torch.amp import autocast

TOTAL = 10 ** 6
BATCH_SIZE = 16

TOPICS = [
    "Conținut pentru adulți",
    "Artă și design",
    "Dezvoltare software",
    "Crime și investigații",
    "Educație și joburi",
    "Electronică și hardware",
    "Divertisment",
    "Viață socială",
    "Modă și frumusețe",
    "Finanțe și afaceri",
    "Mâncare și băuturi",
    "Jocuri",
    "Sănătate",
    "Istorie și geografie",
    "Hobby-uri și casă",
    "Industrial",
    "Literatură",
    "Politică",
    "Religie",
    "Știință, matematică și tehnologie",
    "Software",
    "Sport și fitness",
    "Transport",
    "Turism și călătorii",
]


FORMATS = [
    "Articol academic",
    "Cuprins",
    "Scriere creativă",
    "Pagină de asistență pentru clienți",
    "Forum de discuții",
    "Întrebări frecvente (FAQs)",
    "Conținut incomplet",
    "Articol de cunoștințe",
    "Notificări legale",
    "Articol de tip listă",
    "Articol de știri",
    "Scriere non-ficțiune",
    "Pagină despre organizație",
    "Anunț organizațional",
    "Pagină personală",
    "Blog personal",
    "Pagină de produs",
    "Forum întrebări și răspunsuri",
    "Spam și reclame",
    "Date structurate",
    "Scriere tehnică",
    "Transcriere sau interviu",
    "Tutorial sau ghid",
    "Recenzii ale utilizatorilor",
]


def infer(model1, model2, tokenizer, texts, counts, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad(), autocast("cuda"):
        outputs1 = model1(**inputs)
        outputs2 = model2(**inputs)

    for pred in outputs1.logits.argmax(dim=-1).tolist():
        counts['topic'][pred] += 1

    for pred in outputs2.logits.argmax(dim=-1).tolist():
        counts['format'][pred] += 1

    del inputs, outputs1, outputs2



def main():
    print("Loading models and tokenizers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("WebOrganizer/TopicClassifier-NoURL")
    device = torch.device("cuda")

    model1 = AutoModelForSequenceClassification.from_pretrained(
        "WebOrganizer/TopicClassifier-NoURL",
        trust_remote_code=True,
        use_memory_efficient_attention=False
    )
    model1.eval()
    model1.to(device)
    print("Model1 loaded.", flush=True)

    model2 = AutoModelForSequenceClassification.from_pretrained(
        "WebOrganizer/FormatClassifier-NoURL",
        trust_remote_code=True,
        use_memory_efficient_attention=False
    )
    model2.eval()
    model2.to(device)
    print("Model2 loaded.", flush=True)

    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    batch = []
    counts = {
        'topic': {i: 0 for i in range(len(TOPICS))},
        'format': {i: 0 for i in range(len(FORMATS))}
    }
    for idx, example in tqdm(enumerate(ds), total=TOTAL, desc="Processing examples"):
        if idx >= TOTAL:
            break
        text = example["text"]
        batch.append(text)
        if len(batch) >= BATCH_SIZE:
            infer(model1, model2, tokenizer, batch, counts, device)
            batch.clear()

    if batch:
        infer(model1, model2, tokenizer, batch, counts, device)
    
    counts['topic'] = {TOPICS[i]: count for i, count in counts['topic'].items()}
    counts['format'] = {FORMATS[i]: count for i, count in counts['format'].items()}
    with open("full_dataset_analysis/counts_edu.json", "w") as f:
        json.dump(counts, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()