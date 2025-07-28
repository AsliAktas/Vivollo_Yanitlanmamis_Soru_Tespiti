"""
03_predict_unlabeled.py
-----------------------
• Tamamlanmış kategori tahminleri
  data/train_kategori.csv ve model.joblib kullanır
• Eksik satırları predictions/output CSV’ye yazar

Usage:
    python src/03_predict_unlabeled.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils import clean_text

def main():
    base = Path(__file__).resolve().parent.parent
    # Load data
    df = pd.read_csv(base / 'data' / 'train_kategori.csv')
    # Load model
    m = joblib.load(base / 'models' / 'cat_model' / 'model.joblib')
    encoder, clf = m['encoder'], m['clf']

    # Predict
    ctx = df['context'].apply(clean_text).tolist()
    emb = encoder.encode(ctx, batch_size=64, show_progress_bar=True,
                         device=("cuda" if encoder.device.type=="cuda" else "cpu"))
    df['kategori'] = clf.predict(emb)

    # Save output
    out = base / 'outputs' / 'vivollo_final.csv'
    df.to_csv(out, index=False)
    print(f"✓ Category predictions saved → {out}")

if __name__=='__main__':
    main()

