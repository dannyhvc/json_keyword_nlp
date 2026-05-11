# API Field Mapper with Sentence Transformers

This project uses **sentence embeddings** and **cosine similarity** to automatically detect semantically similar fields across different JSON APIs. It enables schema inference and consolidation from multiple sources — ideal for merging APIs with differing field names but similar meanings.

---

## Features

- Embeds API field names using `sentence-transformers`
- Computes cosine similarity to infer equivalency
- Supports mapping between fields from two or more JSON schemas
- Prints similarity scores and top field matches
- Extensible for full ETL pipelines or API integration

---

## Installation

pip install sentence-transformers torch

Optional (for CUDA acceleration):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

---

## How It Works

1. Loads a pre-trained embedding model (`all-MiniLM-L6-v2`)
2. Encodes each API's keys as semantic vectors
3. Computes cosine similarity between all combinations
4. Outputs the highest-scoring field matches

---

## Example

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

k1 = ["stock_price", "open_time"]
k2 = ["current_price", "market_open"]

emb1 = model.encode(k1, convert_to_tensor=True)
emb2 = model.encode(k2, convert_to_tensor=True)

cos_sim = util.cos_sim(emb1, emb2)

for i, row in enumerate(cos_sim):
    best_match_idx = row.argmax().item()
    best_score = row[best_match_idx].item()
    print(f"{k1[i]} ≈ {k2[best_match_idx]} (score: {best_score:.3f})")
```

Output:
```txt
stock_price ≈ current_price (score: 0.634)
open_time ≈ market_open (score: 0.565)
```

## Future Additions
Multi-way mapping (more than 2 APIs)

Value-based similarity (e.g. using example values)

Web UI for visual schema mapping

Persistent map storage in SQLite or a vector database

---

## 🧰 Dependencies
sentence-transformers: https://www.sbert.net/

torch (PyTorch): https://pytorch.org/
