"""
07_recalc_yanitlandi.py
----------------------
• Recalculates yanitlandi_mi for all user messages:
  using the next-bot-message rule.
• Overwrites vivollo_final.csv.

Usage:
    python src/07_recalc_yanitlandi.py
"""
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    file_in = base/'outputs'/'vivollo_final.csv'
    df = pd.read_csv(file_in, parse_dates=['created_at'])
    # reset all
    df['yanitlandi_mi'] = ''
    for conv, grp in df.groupby('conversation_id'):
        grp = grp.sort_values('created_at')
        idxs = grp.index.tolist()
        for i, idx in enumerate(idxs):
            if grp.at[idx,'sender_type']!='user': continue
            later = idxs[i+1:]
            has_reply = any(df.at[j,'sender_type']=='bot' for j in later)
            df.at[idx,'yanitlandi_mi'] = 'Evet' if has_reply else 'Hayır'
    df.to_csv(file_in, index=False)
    print(f"✓ yanitlandi_mi recalculated → {file_in}")

if __name__=='__main__':
    main()
