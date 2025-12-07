# FineWeb2-Ro

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/collections/OpenLLM-Ro/pretraining-datasets)
[![Paper](https://img.shields.io/badge/arXiv-2511.01090-b31b1b)](https://arxiv.org/abs/2511.01090)

</div>

**FineWeb2-Ro** is a large-scale, high-quality dataset for the Romanian language, with quality and toppic/format annotations for a better understanding of the data used for model training.

---

## 🚀 Quick Start

You can load the dataset immediately using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("OpenLLM-Ro/fineweb2-ro-bert", split="train")