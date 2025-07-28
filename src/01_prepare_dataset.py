
# ------------------------------
# File: src/01_prepare_dataset.py
# ------------------------------
# -*- coding: utf-8 -*-
"""
01_prepare_dataset.py
---------------------
• data/full_messages.* → data/train_kategori.csv
  - Sadece USER + etiketli
  - context = önceki + mevcut mesaj
  - Nadir sınıflar "Diğer"e merge
"""
import pandas as pd
from pathlib import Path
from utils import clean_text

DATA_DIR  = Path("data")
PARQ_FILE = DATA_DIR / "full_messages.parquet"
CSV_FILE  = DATA_DIR / "full_messages.csv"
OUT_FILE  = DATA_DIR / "train_kategori.csv"

def load_data():
    if PARQ_FILE.exists():
        return pd.read_parquet(PARQ_FILE)
    return pd.read_csv(CSV_FILE)

def main():
    df = load_data()
    mask = (df['sender_type']=='user') & df['kategori'].notna()
    df = df.loc[mask].reset_index(drop=True)
    df['prev_text'] = (df.groupby('conversation_id')['text'].shift(1).fillna(""))
    df['context'] = (df['prev_text'] + ' ' + df['text']).apply(clean_text)
    counts = df['kategori'].value_counts()
    rare = counts[counts<50].index.tolist()
    df['kategori'] = df['kategori'].apply(lambda x: 'Diğer' if x in rare else x)
    # Typo merge
    MERGE_MAP = {}
    df['kategori'] = df['kategori'].map(lambda x: MERGE_MAP.get(x.strip(), x.strip()))
    df[['context','kategori']].to_csv(OUT_FILE, index=False)
    print(f"✓  train_kategori.csv oluşturuldu → {len(df):,} satır")

if __name__=='__main__':
    main()

