# Kirundi News Topic Classification

NLP Summative Project, African Leadership University. Text classification for Kirundi, an under-resourced African language.

## Links
- **Live app:** https://kirundi-news-classifier.streamlit.app/
- **Fine-tuned model:** https://huggingface.co/sonia250/afriberta-kirundi-news
- **Demo video:** (add link)
- **Report:** (add link or see submission)

## Problem
Classify Kirundi news articles into six topics: business, entertainment, health, politics, religion, sports.

## Dataset
MasakhaNEWS, Kirundi subset (`run`), loaded from Hugging Face (`masakhane/masakhanews`).
- 1,117 train / 159 validation / 322 test articles
- Input = headline + article text
- Classes are imbalanced (politics 31% of train, business and religion under 5% each), so macro-F1 is the main metric.

## Methods
| # | Model | Role |
|---|---|---|
| 1 | TF-IDF + Logistic Regression | Baseline |
| 2 | Word2Vec (trained on train text) + BiLSTM | Course technique |
| 3 | AfriBERTa (`castorini/afriberta_base`), fine-tuned | Pretrained African-language model |
| 4 | AfroXLMR (`Davlan/afro-xlmr-base`), fine-tuned | Second pretrained model |

All models use class weights. Transformers: max length 512 tokens, lr 3e-5 (AfriBERTa) and 2e-5 (AfroXLMR), batch size 8, 10 epochs, best epoch chosen by validation macro-F1, seed 42.

## Results (test set, 322 articles)
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.863 | 0.776 | 0.859 |
| Word2Vec + BiLSTM | 0.839 | 0.744 | 0.835 |
| **AfriBERTa (fine-tuned)** | **0.916** | **0.853** | **0.915** |
| AfroXLMR (fine-tuned) | 0.901 | 0.829 | 0.899 |

Error analysis (AfriBERTa, 27 of 322 errors): business and entertainment are the weakest classes; most errors are topic overlap (e.g. politics vs business) and short texts. 21 of the 27 errors were also made by AfroXLMR. Details: `results/afriberta_errors.csv` and the report.

## Repository structure
- `notebooks/`: data exploration, baselines (TF-IDF, BiLSTM), AfriBERTa, AfroXLMR
- `src/`: reproducible scripts (`data.py`, `baseline_tfidf.py`, `finetune.py`, `predict.py`)
- `app/app.py`: Streamlit web app (loads the model from Hugging Face)
- `results/`: figures, metrics, predictions, error list

## How to run
```bash
git clone https://github.com/sonia250/kirundi-news-classifier
cd kirundi-news-classifier
pip install -r requirements-train.txt

python src/baseline_tfidf.py                       # baseline
python src/finetune.py --model castorini/afriberta_base --lr 3e-5 --out afriberta_kirundi   # GPU recommended
python src/predict.py "Kirundi headline. Article text ..."   # uses the published model

pip install -r requirements.txt && streamlit run app/app.py   # run the app locally
```
The BiLSTM experiment is in `notebooks/02_baselines.ipynb`.

## Limitations
Small dataset (1,117 training articles); small test classes (16 business, 15 religion), so scores on them are noisy; each model was run once; texts are cut at 512 tokens; no "other" class, so out-of-topic text is still assigned one of the six topics; short headlines alone are unreliable.

## Acknowledgements and licenses
- MasakhaNEWS dataset (Adelani et al., 2023), Masakhane
- AfriBERTa (Ogueji et al., 2021), AfroXLMR (Alabi et al., 2022), Hugging Face Transformers, scikit-learn, gensim, TensorFlow, Streamlit
- Code released under the MIT License.