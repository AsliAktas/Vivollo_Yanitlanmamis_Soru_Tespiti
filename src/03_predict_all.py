"""
03_predict_all.py
-----------------
• Tüm user mesajlarına kategori, sentiment, intent tahmini yapar
• models/*_model/model.joblib kullanır
• Çıktı: outputs/vivollo_final.csv

Usage:
    python src/03_predict_all.py
"""
import pandas as pd
import joblib
from pathlib import Path
from utils import clean_text


def main():
    base = Path(__file__).resolve().parent.parent
    # Load full messages
    file = base / 'data' / 'full_messages.csv'
    df = pd.read_csv(file)
    mask = df['sender_type']=='user'

    # Prepare context
    df['prev_text'] = df.groupby('conversation_id')['text'].shift(1).fillna('')
    ctx = (df.loc[mask,'prev_text'] + ' ' + df.loc[mask,'text']).apply(clean_text).tolist()

    # Load and predict category
    cat = joblib.load(base/'models'/'cat_model'/'model.joblib')
    emb = cat['encoder'].encode(ctx, batch_size=64, show_progress_bar=True)
    df.loc[mask,'kategori'] = cat['clf'].predict(emb)

    # Load and predict sentiment
    sent = joblib.load(base/'models'/'sentiment_model'/'model.joblib')
    emb = sent['encoder'].encode(ctx, batch_size=64, show_progress_bar=True)
    df.loc[mask,'sentiment'] = sent['clf'].predict(emb)

    # Load and predict intent
    intent = joblib.load(base/'models'/'intent_model'/'model.joblib')
    emb = intent['encoder'].encode(ctx, batch_size=64, show_progress_bar=True)
    df.loc[mask,'intent'] = intent['clf'].predict(emb)

    # Save
    out = base / 'outputs' / 'vivollo_final.csv'
    df.to_csv(out, index=False)
    print(f"✓ All predictions saved → {out}")

if __name__=='__main__':
    main()

