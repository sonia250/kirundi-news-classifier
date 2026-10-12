"""Load MasakhaNEWS (Kirundi) and build the model input text."""
import pandas as pd
from datasets import load_dataset


def load_splits():
    ds = load_dataset("masakhane/masakhanews", "run")
    out = {}
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(ds[split])
        df["input"] = df["headline"].fillna("") + ". " + df["text"].fillna("")
        out[split] = df
    return out