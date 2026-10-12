"""Baseline: TF-IDF + Logistic Regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data import load_splits

splits = load_splits()
train, val, test = splits["train"], splits["validation"], splits["test"]

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                      sublinear_tf=True, min_df=2)
Xtr = vec.fit_transform(train["input"])
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, train["category"])

for name, df in [("VALIDATION", val), ("TEST", test)]:
    pred = clf.predict(vec.transform(df["input"]))
    print(name, "acc %.3f  macro-F1 %.3f" % (
        accuracy_score(df["category"], pred),
        f1_score(df["category"], pred, average="macro")))

print(classification_report(
    test["category"], clf.predict(vec.transform(test["input"])), digits=3))