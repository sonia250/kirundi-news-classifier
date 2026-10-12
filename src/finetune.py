"""Fine-tune a pretrained Transformer (AfriBERTa or AfroXLMR) for topic classification.

Usage:
  python src/finetune.py --model castorini/afriberta_base --lr 3e-5 --out afriberta_kirundi
  python src/finetune.py --model Davlan/afro-xlmr-base --lr 2e-5 --out afroxlmr_kirundi
"""
import argparse
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          set_seed)
from data import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="castorini/afriberta_base")
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--out", default="model_out")
args = parser.parse_args()

set_seed(42)
splits = load_splits()
train, val, test = splits["train"], splits["validation"], splits["test"]
labels = sorted(train["category"].unique())
l2i = {l: i for i, l in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained(args.model)


def make_ds(df):
    enc = tokenizer(df["input"].tolist(), truncation=True, max_length=512)
    return Dataset.from_dict({**enc, "labels": [l2i[c] for c in df["category"]]})


tr_ds, va_ds, te_ds = make_ds(train), make_ds(val), make_ds(test)

ytr = np.array([l2i[c] for c in train["category"]])
cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(labels)),
                                       y=ytr), dtype=torch.float)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        y = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, y, weight=cw.to(out.logits.device))
        return (loss, out) if return_outputs else loss


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}


model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels),
    id2label={i: l for l, i in l2i.items()}, label2id=l2i)

targs = TrainingArguments(
    output_dir=args.out + "_tmp", learning_rate=args.lr,
    per_device_train_batch_size=8, per_device_eval_batch_size=16,
    num_train_epochs=args.epochs, weight_decay=0.01, warmup_steps=140,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", save_total_limit=1,
    fp16=torch.cuda.is_available(), seed=42, report_to="none")

trainer = WeightedTrainer(model=model, args=targs, train_dataset=tr_ds,
                          eval_dataset=va_ds,
                          data_collator=DataCollatorWithPadding(tokenizer),
                          compute_metrics=compute_metrics)
trainer.train()

out = trainer.predict(te_ds)
y_true, y_pred = out.label_ids, out.predictions.argmax(-1)
print("TEST acc %.3f  macro-F1 %.3f" % (
    accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))
print(classification_report(y_true, y_pred, target_names=labels, digits=3))

trainer.save_model(args.out)
tokenizer.save_pretrained(args.out)