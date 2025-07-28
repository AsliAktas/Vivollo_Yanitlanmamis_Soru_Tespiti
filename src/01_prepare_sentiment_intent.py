"""
01_prepare_sentiment_intent.py
------------------------------
• data/full_messages.* →
  - data/train_sentiment.csv (context, sentiment)
  - data/train_intent.csv    (context, intent)
"""
import pandas as pd
from pathlib import Path
from utils import clean_text

DATA = Path("data")
PARQ = DATA / "full_messages.parquet"
CSV  = DATA / "full_messages.csv"

df = pd.read_parquet(PARQ) if PARQ.exists() else pd.read_csv(CSV)
mask = (df.sender_type=='user') & df.sentiment.notna() & df.intent.notna()
df = df.loc[mask].reset_index(drop=True)
df['prev_text'] = df.groupby('conversation_id')['text'].shift(1).fillna("")
df['context']   = (df['prev_text'] + ' ' + df['text']).apply(clean_text)

df[['context','sentiment']].to_csv(DATA/'train_sentiment.csv', index=False)
df[['context','intent']].to_csv(DATA/'train_intent.csv', index=False)
print(f"✓ train_sentiment.csv → {len(df)} satır")
print(f"✓ train_intent.csv   → {len(df)} satır")
