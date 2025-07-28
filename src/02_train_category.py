# -*- coding: utf-8 -*-
"""
02_train_category.py
--------------------
• data/train_kategori.csv → kategori modeli
• Kaydet: models/cat_model/model.joblib

Usage:
    python src/02_train_category.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import RandomOverSampler

# Settings
DATA_DIR     = Path("data")
TRAIN_FILE   = DATA_DIR / "train_kategori.csv"
MODEL_DIR    = Path("models/cat_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def main():
    # 1) Load training data
    df = pd.read_csv(TRAIN_FILE)
    X_text = df["context"].astype(str).tolist()
    y       = df["kategori"]

    # 2) Encode
    encoder = SentenceTransformer(ENCODER_NAME)
    print("🔎 Encoding category texts...")
    X_emb = encoder.encode(
        X_text, batch_size=64, show_progress_bar=True,
        device=("cuda" if encoder.device.type=="cuda" else "cpu")
    )

    # 3) Balance classes
    ros = RandomOverSampler()
    X_res, y_res = ros.fit_resample(X_emb, y)

    # 4) Train with cross‑validation
    print("⚙️ Training category model...")
    clf = LogisticRegressionCV(
        Cs=[0.2, 0.5, 1.0],
        cv=3,
        max_iter=4000,
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        scoring="f1_macro"
    )
    clf.fit(X_res, y_res)

    # 5) Save
    joblib.dump({"encoder": encoder, "clf": clf}, MODEL_DIR / "model.joblib")
    print(f"✓ Category model saved → {MODEL_DIR/'model.joblib'}")

if __name__ == "__main__":
    main()
