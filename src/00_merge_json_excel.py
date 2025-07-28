"""
00_merge_json_excel.py
----------------------
Combines fully labeled Excel with new JSON entries by matching common ID column.
- Supports arbitrary JSON key for ID (e.g., message_id, messageId, id).
- Appends only missing entries.
- Outputs to data/full_messages.csv or parquet.

Usage:
    python src/00_merge_json_excel.py
"""
import json
import pandas as pd
from pathlib import Path
import re
import warnings

def guess_id_column(cols):
    """Return best guess for ID column: message_id, id, otherwise first column."""
    # Ensure string names
    col_names = [str(c) for c in cols]
    # Try message_id
    for orig, col in zip(cols, col_names):
        if re.search(r'message[_]?id$', col, re.IGNORECASE):
            return orig
    # Try 'id'
    for orig, col in zip(cols, col_names):
        if col.lower() == 'id':
            return orig
    # fallback to first
    warnings.warn(f"No explicit ID column found in {col_names}, using first column '{col_names[0]}'")
    return cols[0]


def main():
    base = Path(__file__).resolve().parent.parent
    # Load final fully labeled Excel
    final_excel = base / 'outputs' / 'vivollo_final.xlsx'
    if not final_excel.exists():
        raise FileNotFoundError(f"Tam etiketli dosya bulunamadı: {final_excel}")
    df_final = pd.read_excel(final_excel)
    final_id = guess_id_column(df_final.columns)

    # Load JSON comments
    json_file = base / 'data' / 'yorumlar.json'
    if not json_file.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {json_file}")
    with open(json_file, encoding='utf-8') as f:
        data_json = json.load(f)
    # Normalize nested messages list to flat rows
    # Each element in data_json assumed to have 'conversation_id' and 'messages' list
    try:
        df_json = pd.json_normalize(data_json, record_path=['messages'], meta=['conversation_id'])
    except Exception:
        # Fallback: top-level normalize then expand
        df_json = pd.json_normalize(data_json)
        if 'messages' in df_json:
            msg_expanded = pd.json_normalize(df_json['messages'])
            df_json = df_json.drop(columns=['messages']).join(msg_expanded)
    # Now df_json should have 'message_id' column
    json_id = guess_id_column(df_json.columns)
    print(f"Using JSON ID column: '{json_id}' and Excel ID column: '{final_id}'")

    # Ensure both IDs present
    if final_id not in df_final or json_id not in df_json:
        raise KeyError(f"ID column mismatch: final has {final_id}, json has {json_id}")

    # Find missing entries
    mask = ~df_json[json_id].isin(df_final[final_id])
    df_missing = df_json.loc[mask]
    print(f"➕ JSON’dan eklenen yeni satır: {len(df_missing)}")

    # Align ID column names
    if json_id != final_id:
        df_missing = df_missing.rename(columns={json_id: final_id})

    # Concatenate
    df_full = pd.concat([df_final, df_missing], ignore_index=True)

    # Save parquet or CSV
    out_parquet = base / 'data' / 'full_messages.parquet'
    try:
        df_full.to_parquet(out_parquet, index=False)
        print(f"✓ full_messages.parquet oluşturuldu → {out_parquet}")
    except Exception:
        out_csv = base / 'data' / 'full_messages.csv'
        df_full.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"⚠️ full_messages.csv oluşturuldu → {out_csv}")

if __name__ == '__main__':
    main()
