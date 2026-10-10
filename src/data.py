import json
import os
import random
from datasets import load_dataset

SEED = 42
N_DEV = 500
N_TEST = 2000

ds = load_dataset("rajpurkar/squad")

text_to_id = {}
corpus = []
for split in ["train", "validation"]:
    for row in ds[split]:
        c = row["context"]
        if c not in text_to_id:
            text_to_id[c] = len(corpus)
            corpus.append(c)

val_rows = [
    {
        "id": row["id"],
        "question": row["question"],
        "context_id": text_to_id[row["context"]],
        "answer": row["answers"]["text"][0],
    }
    for row in ds["validation"]
]

rng = random.Random(SEED)
rng.shuffle(val_rows)
dev = val_rows[:N_DEV]
test = val_rows[N_DEV:N_DEV + N_TEST]

assert not ({r["id"] for r in dev} & {r["id"] for r in test})

os.makedirs("data", exist_ok=True)
with open("data/corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f)
with open("data/dev.json", "w", encoding="utf-8") as f:
    json.dump(dev, f)
with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(test, f)

words = [len(c.split()) for c in corpus]
print("paragraphs in corpus:", len(corpus))
print("mean words per paragraph:", round(sum(words) / len(words), 1))
print("dev questions:", len(dev))
print("test questions:", len(test))
print("distinct target paragraphs in test:", len({r["context_id"] for r in test}))
print("example question:", test[0]["question"])
print("its answer:", test[0]["answer"])
