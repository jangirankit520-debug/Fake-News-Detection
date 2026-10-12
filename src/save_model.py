import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

from clean_data import load_raw_data, basic_clean, remove_leakage_clues
from src.clean_data import project_root
from src.train_baseline import vectorizer

project_root=Path(__file__).resolve().parent.parent
models_dir=project_root/"models"
models_dir.mkdir(exist_ok=True)

data=load_raw_data()
data=basic_clean(data)
data["text_no_leak"]=data["text"].apply(remove_leakage_clues)

x=data["text_no_leak"]
y=data["label"]
vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
xv=vectorizer.fit_transform(x)

model=PassiveAggressiveClassifier(max_iter=50,random_state=42)
model.fit(xv,y)

joblib.dump(model,models_dir/"model.joblib")
joblib.dump(vectorizer,models_dir/"vectorizer.joblib")
print("Model and vectorizer saved to:",models_dir)








