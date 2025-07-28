"""
04_format_output.py
-------------------
• outputs/vivollo_final.csv → outputs/vivollo_final_clean.xlsx
• Temiz bir preview sheet oluşturur

Usage:
    python src/04_format_output.py
"""
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base/'outputs'/'vivollo_final.csv')

    # Preview formatting
    df_preview = df[['conversation_id','message_id','created_at',
                     'sender_type','text','yanitlandi_mi',
                     'sentiment','kategori','intent']]

    out_xl = base/'outputs'/'vivollo_final_clean.xlsx'
    with pd.ExcelWriter(out_xl, engine='openpyxl') as writer:
        df_preview.to_excel(writer, sheet_name='preview', index=False)
    print(f"✓ Clean Excel created → {out_xl}")

if __name__=='__main__':
    main()
