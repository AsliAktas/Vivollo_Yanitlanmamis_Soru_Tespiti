"""
05_split_safe.py
---------------
• Temporal / random split for train, val, test sets
• Ensures a minimum test size, falls back to final 10%

Usage:
    python src/05_split_safe.py
"""
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base/'data'/'train_kategori.csv')

    # Temporal split by created_at (if available)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        # e.g., last quarter as test
        cutoff = df['created_at'].quantile(0.9)
        test = df[df['created_at'] > cutoff]
    else:
        test = pd.DataFrame()

    # Ensure min test size
    if test.shape[0] < 1:
        test = df.sample(frac=0.1, random_state=42)

    remaining = df.drop(test.index)
    val = remaining.sample(frac=0.2, random_state=42)
    train = remaining.drop(val.index)

    # Save
    split_dir = base/'data'
    train.to_csv(split_dir/'train.csv', index=False)
    val.to_csv(split_dir/'val.csv', index=False)
    test.to_csv(split_dir/'test.csv', index=False)
    print(f"✓ Split done – train:{train.shape[0]}, val:{val.shape[0]}, test:{test.shape[0]}")

if __name__=='__main__':
    main()

