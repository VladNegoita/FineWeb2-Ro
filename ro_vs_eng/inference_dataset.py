import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from torch.amp import autocast
from argparse import ArgumentParser
import torch.multiprocessing as mp


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

TOPICS_TO_ENG = {
    "Conținut pentru adulți": "Adult content",
    "Artă și design": "Art & Design",
    "Dezvoltare software": "Software development",
    "Crime și investigații": "Crime & Law",
    "Educație și joburi": "Education & Jobs",
    "Electronică și hardware": "Electronics & Hardware",
    "Divertisment": "Entertainment",
    "Viață socială": "Social Life",
    "Modă și frumusețe": "Fashion & Beauty",
    "Finanțe și afaceri": "Finance & Business",
    "Mâncare și băuturi": "Food & Dining",
    "Jocuri": "Games",
    "Sănătate": "Health",
    "Istorie și geografie": "History & Geography",
    "Hobby-uri și casă": "Hobbies & Home",
    "Industrial": "Industrial",
    "Literatură": "Literature",
    "Politică": "Politics",
    "Religie": "Religion",
    "Știință, matematică și tehnologie": "Science, Math & Tech",
    "Software": "Software",
    "Sport și fitness": "Sports & Fitness",
    "Transport": "Transportation",
    "Turism și călătorii": "Travel & Tourism",
}

FORMATS_TO_ENG = {
    "Articol academic": "Academic Writing",
    "Cuprins": "Content Listing",
    "Scriere creativă": "Creative Writing",
    "Pagină de asistență pentru clienți": "Customer Support Page",
    "Forum de discuții": "Discussion Forum",
    "Întrebări frecvente (FAQs)": "FAQs",
    "Conținut incomplet": "Incomplete Content",
    "Articol de cunoștințe": "Knowledge Article",
    "Notificări legale": "Legal Notices",
    "Articol de tip listă": "Listicle",
    "Articol de știri": "News Article",
    "Scriere non-ficțiune": "Nonfiction Writing",
    "Pagină despre organizație": "Org. About Page",
    "Anunț organizațional": "Org. Announcement",
    "Pagină personală": "Personal About Page",
    "Blog personal": "Personal Blog",
    "Pagină de produs": "Product Page",
    "Forum întrebări și răspunsuri": "Q&A Forum",
    "Spam și reclame": "Spam / Ads",
    "Date structurate": "Structured Data",
    "Scriere tehnică": "Technical Writing",
    "Transcriere sau interviu": "Transcript / Interview",
    "Tutorial sau ghid": "Tutorial",
    "Recenzii ale utilizatorilor": "User Reviews",
}

BATCH_SIZE = 16

def annotate(model1, model2, tokenizer, examples, device):
    for start_idx in range(0, len(examples), BATCH_SIZE):
        batch = examples[start_idx:start_idx + BATCH_SIZE]
        inputs = tokenizer([ex["text"] for ex in batch], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad(), autocast("cuda"):
            outputs1 = model1(**inputs)
            outputs2 = model2(**inputs)

        for i, example in enumerate(batch):
            example["topic_class_1"] = TOPICS_TO_ENG[TOPICS[outputs1.logits.argmax(dim=-1)[i].item()]]
            example["format_class_1"] = FORMATS_TO_ENG[FORMATS[outputs2.logits.argmax(dim=-1)[i].item()]]

        del inputs, outputs1, outputs2



def worker(rank: int, base_path: str, dataset: str):
    print(f"Worker {rank} started for dataset {dataset}...", flush=True)

    # Load models and tokenizers
    print("Loading models and tokenizers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("WebOrganizer/TopicClassifier-NoURL")
    device = torch.device(f"cuda:{rank}")

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

    in_dir = os.path.join(base_path, dataset)
    out_dir = os.path.join(base_path, dataset + "_annotated")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(in_dir):
        raise ValueError(f"Input directory {in_dir} does not exist.")

    curr_file_idx = rank
    while True:
        curr_file = os.path.join(in_dir, f"{dataset}_{curr_file_idx}.json")
        if not os.path.exists(curr_file):
            print(f"File {curr_file} does not exist. Stopping.")
            break

        if os.path.exists(os.path.join(out_dir, f"{dataset}_{curr_file_idx}_annotated.json")):
            print(f"Annotated file {curr_file} already exists. Skipping.", flush=True)
            curr_file_idx += 2
            continue

        with open(curr_file, "r") as f:
            examples = json.load(f)

        print(f"Processing file {curr_file} with {len(examples)} examples.", flush=True)
        annotate(model1, model2, tokenizer, examples, device)
        with open(os.path.join(out_dir, f"{dataset}_{curr_file_idx}_annotated.json"), "w") as f:
            json.dump(examples, f, indent=4, ensure_ascii=False)

        curr_file_idx += 2
        print(f"Finished processing file {curr_file}.", flush=True)

def main(base_path: str, dataset: str):
    print(f"Starting processing for dataset {dataset} in {base_path}...", flush=True)
    mp.spawn(worker, args=(base_path, dataset), nprocs=2, join=True)
    print("Processing completed.", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args.base_path, args.dataset)