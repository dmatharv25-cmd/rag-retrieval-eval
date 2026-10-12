import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort
from sentence_transformers import SentenceTransformer

tk = Tokenizer.from_file("minilm_onnx/tokenizer.json")
tk.enable_truncation(max_length=256)
tk.enable_padding()
sess = ort.InferenceSession("minilm_onnx/model.onnx", providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
print("onnx inputs:", sorted(names))


def embed(texts):
    encs = tk.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.zeros_like(ids)
    out = sess.run(None, feed)[0]
    v = (out * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


qs = ["Who was the first emperor of Rome?", "What year did the Normans conquer England?", "What is septicemia?"]
a = embed(qs)
b = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(qs, normalize_embeddings=True)
print("cosine ONNX vs torch:", [round(float(x @ y), 5) for x, y in zip(a, b)])
