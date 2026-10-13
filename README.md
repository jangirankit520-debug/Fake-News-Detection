# 📰 Fake News Detection

A machine learning project that classifies news articles as real or fake — and investigates *why* the initial 99% accuracy was misleading.

## The problem

Standard fake-news datasets (like this one) often contain **leakage**: patterns unrelated to truth that let a model "cheat." This project doesn't just train a classifier, it hunts down and measures that leakage.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, ISOT), two CSVs: `Fake.csv` (23,481 articles) and `True.csv` (21,417 articles). Not included in this repo due to size — download from the link above and place in a `Data/` folder.

## Approach

1. **EDA** — checked class balance, duplicates, missing values, and compared `subject` tags between the two files.
2. **Cleaning** — removed 6,252 duplicate articles and 631 empty rows.
3. **Baseline models** — compared Logistic Regression, Naive Bayes, and Passive Aggressive Classifier using TF-IDF features.
4. **Leakage investigation** — discovered real articles nearly always start with a dateline + "(Reuters)" tag, and a model trained on raw text can exploit this instead of judging content. Iteratively removed leakage clues (publisher tags, weekday datelines, web/photo-credit artifacts) and re-measured.
5. **Error analysis** — inspected top TF-IDF feature weights and misclassified articles to find further leakage sources.
6. **Deployment** — saved the final model with `joblib` and built a Streamlit demo for live predictions.

## Results

| Version | Accuracy | F1 Score |
|---|---|---|
| With full leakage (raw text) | 99.06% | 99.14% |
| After removing dateline/publisher tags | 98.11% | 98.28% |
| After also removing weekday/web artifacts | 97.63% | 97.84% |

The ~1.4 point drop shows part of the original accuracy came from stylistic shortcuts, not genuine content understanding. The remaining 97.6% reflects a more honest, content-based performance — though some residual leakage (e.g. publisher names) likely remains, which a production system would need to handle more systematically (e.g. named-entity removal).

## Key finding

A model trained on this kind of dataset learns **writing style**, not truth. It cannot verify facts, and performance on real, current news (different time period, different publishers) would likely be much lower. See [Limitations](#limitations).

## Limitations

- Trained only on 2016–2017 US political news; doesn't generalize to other time periods, topics, or publishers.
- Detects stylistic patterns, not factual accuracy — it cannot verify claims against real-world evidence.
- Residual leakage (e.g. publisher names, photo-credit sites) likely remains despite cleaning.

## Project structure
fake-news-detection/
├── Data/ # Fake.csv, True.csv (not included, see above)
├── src/
│ ├── clean_data.py # loading, cleaning, leakage removal
│ ├── explore_data.py # EDA script
│ ├── train_baseline.py # model comparison
│ ├── train_no_leak.py # leakage experiment
│ ├── analyze_errors.py # top features + error analysis
│ └── save_model.py # trains final model, saves to /models
├── models/ # saved model + vectorizer (generated, not committed)
├── app.py # Streamlit demo
├── requirements.txt
└── README.md


## How to run

```bash
pip install -r requirements.txt

# Regenerate the model (run once)
python src/save_model.py

# Launch the demo
streamlit run app.py
```

## Tech stack

Python · pandas · scikit-learn · TF-IDF · Streamlit · joblib