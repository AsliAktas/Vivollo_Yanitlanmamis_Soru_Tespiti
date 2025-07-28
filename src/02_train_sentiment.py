# -*- coding: utf-8 -*-
"""
02_train_sentiment.py
---------------------
• data/train_sentiment.csv → sentiment modeli
• Kaydet: models/sentiment_model/model.joblib

Usage:
    python src/02_train_sentiment.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from utils import clean_text

DATA_DIR  = Path("data")
TRAIN_CSV = DATA_DIR / "train_sentiment.csv"
OUT_DIR   = Path("models/sentiment_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODER   = "paraphrase-multilingual-MiniLM-L12-v2"

def main():
    # 1) Load data
    df = pd.read_csv(TRAIN_CSV)
    X = df["context"].apply(clean_text).tolist()
    y = df["sentiment"]

    # 2) Embed
    encoder = SentenceTransformer(ENCODER)
    print("🔎 Encoding sentiment texts...")
    emb = encoder.encode(
        X, batch_size=64, show_progress_bar=True,
        device=("cuda" if encoder.device.type=="cuda" else "cpu")
    )

    # 3) Train (single C value)
    print("⚙️ Training sentiment model...")
    clf = LogisticRegression(
        C=0.5,
        max_iter=4000,
        class_weight="balanced",
        multi_class="multinomial",
        n_jobs=-1
    )
    clf.fit(emb, y)

    # 4) Save
    joblib.dump({"encoder": encoder, "clf": clf}, OUT_DIR / "model.joblib")
    print(f"✓ Sentiment model saved → {OUT_DIR/'model.joblib'}")

if __name__ == "__main__":
    main()
