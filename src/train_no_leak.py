from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from clean_data import load_raw_data, basic_clean, remove_leakage_clues
from src.train_baseline import vectorizer, xv_train

data=load_raw_data()
data=basic_clean(data)
data["text_no_leak"]=data["text"].apply(remove_leakage_clues)

def train_and_evaluate(x,y,label):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y )
    vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
    xv_train=vectorizer.fit_transform(x_train)
    xv_test=vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(xv_train, y_train)
    y_pred = model.predict(xv_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- {label} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")

    return acc

y=data["label"]
acc_with_leak=train_and_evaluate(data["text"], y, "With leakage (original text)")
acc_no_leak = train_and_evaluate(data["text_no_leak"], y, "Without leakage (cleaned text)")

print(f"\nAccuracy dropped by: {(acc_with_leak - acc_no_leak) * 100:.2f} percentage points")











