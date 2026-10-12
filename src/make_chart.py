import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# recall@1 on the 2,000-question SQuAD test set, copied from the README table
rows = [
    ("MiniLM", 0.527),
    ("BGE-small", 0.600),
    ("BM25", 0.683),
    ("Hybrid (MiniLM)", 0.731),
    ("Hybrid (BGE-small)", 0.750),
    ("Hybrid (BGE-small)\n+ rerank top 20", 0.830),
]
names = [r[0] for r in rows]
vals = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(names, vals, color=["#9aa5b1"] * 3 + ["#5b8def"] * 2 + ["#1f4fd8"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("recall@1")
ax.set_title("SQuAD test (2,000 questions): recall@1 by method")
plt.xticks(fontsize=8)
plt.tight_layout()
plt.savefig("docs/recall_at_1.png", dpi=150)
print("saved docs/recall_at_1.png")
