"""Predict the topic of a Kirundi news text with the fine-tuned model.

Usage:  python src/predict.py "headline. article text ..."
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "sonia250/afriberta-kirundi-news"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

text = " ".join(sys.argv[1:])
inputs = tok(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
for i in probs.argsort(descending=True):
    print("%-14s %.3f" % (model.config.id2label[int(i)], float(probs[i])))