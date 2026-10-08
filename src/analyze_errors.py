import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from clean_data import load_raw_data, basic_clean, remove_leakage_clues

data = load_raw_data()
data = basic_clean(data)
data["text_no_leak"] = data["text"].apply(remove_leakage_clues)

x = data["text_no_leak"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(xv_train, y_train)

feature_names=vectorizer.get_feature_names_out()
coefficients=model.coef_[0]
word_importance=pd.DataFrame({"word":feature_names,"weight":coefficients})
word_importance=word_importance.sort_values("weight")

print("Top 15 words pushing toward FAKE:")
print(word_importance.head(15).to_string(index=False))

print("\nTop 15 words pushing toward REAL:")
print(word_importance.tail(15).to_string(index=False))

y_pred=model.predict(xv_test)

result_df=pd.DataFrame({"text":x_test.values,"actual":y_test.values,"predicted":y_pred})

errors = result_df[result_df["actual"] != result_df["predicted"]]
print(f"\nTotal errors: {len(errors)} out of {len(result_df)}")

print("\nSample misclassified articles:")
for i, row in errors.head(5).iterrows():
    actual_label = "REAL" if row["actual"] == 1 else "FAKE"
    predicted_label = "REAL" if row["predicted"] == 1 else "FAKE"
    print(f"\nActual: {actual_label} | Predicted: {predicted_label}")
    print(row["text"][:200])


















