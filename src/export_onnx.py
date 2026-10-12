import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer

NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT = sys.argv[1] if len(sys.argv) > 1 else "minilm_onnx"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(NAME)
tok.save_pretrained(OUT)
model = AutoModel.from_pretrained(NAME, attn_implementation="eager").eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


enc = tok(["hello world", "a longer example sentence here"], return_tensors="pt", padding=True)
torch.onnx.export(
    Wrap(model), (enc["input_ids"], enc["attention_mask"]), f"{OUT}/model.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "last_hidden_state": {0: "b", 1: "s"}},
    opset_version=17, dynamo=False,
)
print("exported", os.path.getsize(f"{OUT}/model.onnx"), "bytes to", OUT)
