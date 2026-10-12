import streamlit as st
import joblib
from pathlib import Path

from src.clean_data import remove_leakage_clues

project_root = Path(__file__).resolve().parent
models_dir = project_root / "models"

model = joblib.load(models_dir / "model.joblib")
vectorizer = joblib.load(models_dir / "vectorizer.joblib")

st.title("📰 Fake News Detector")
st.write(
    "Paste a news article below. The model predicts REAL or FAKE based on "
    "writing style and content patterns learned from a 2016-2017 news dataset."
)

user_input = st.text_area("Article text:", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please paste some text first.")
    else:
        cleaned = remove_leakage_clues(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Predicted: REAL")
        else:
            st.error("🚨 Predicted: FAKE")

        st.caption(
            "Note: this model was trained on 2016-2017 US political news and "
            "detects writing style, not verified truth. Treat predictions as "
            "exploratory, not fact-checking."
        )