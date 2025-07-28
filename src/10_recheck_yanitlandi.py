"""
10_recheck_yanitlandi.py
------------------------
Revalidates and corrects the 'yanitlandi_mi' column for user messages:
- Only processes user rows.
- In manual (green) rows, fixes any mislabels.
- In new (blue) rows, fills in the correct 'Evet'/'Hayır'.
- Uses conversation_id & chronological order.

Output: outputs/vivollo_final_checked.csv

Usage:
    python src/10_recheck_yanitlandi.py
"""
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    # 1) Load the combined CSV
    csv_in = base / "outputs" / "vivollo_final.csv"
    df = pd.read_csv(csv_in, parse_dates=["created_at"])

    # 2) Load manual IDs
    data_dir = base / "data"
    manual_file = next(data_dir.glob("vivollo_clean*.xlsx"), None)
    if not manual_file:
        raise FileNotFoundError("Manual file not found (vivollo_clean*.xlsx)")
    df_manual = pd.read_excel(manual_file)
    manual_ids = set(
        df_manual.loc[df_manual["sender_type"]=="user", "message_id"]
    )

    corrected = []

    # 3) Recalculate per conversation
    for conv_id, grp in df.groupby("conversation_id", sort=False):
        sub = grp.sort_values("created_at")
        # List of bot reply times
        bots = sub.loc[sub["sender_type"]=="bot", "created_at"].tolist()

        for idx, row in sub.iterrows():
            if row["sender_type"] != "user":
                continue

            # Determine expected label
            has_reply = any(bt > row["created_at"] for bt in bots)
            expected = "Evet" if has_reply else "Hayır"

            # If manual row, fix mislabels; if new row, fill missing
            if row["message_id"] in manual_ids:
                if row["yanitlandi_mi"] != expected:
                    df.at[idx, "yanitlandi_mi"] = expected
                    corrected.append((row["message_id"], row["yanitlandi_mi"], expected))
            else:
                if row["yanitlandi_mi"] != expected:
                    df.at[idx, "yanitlandi_mi"] = expected
                    corrected.append((row["message_id"], row["yanitlandi_mi"], expected))

    # 4) Save corrected CSV
    out_csv = base / "outputs" / "vivollo_final_checked.csv"
    df.to_csv(out_csv, index=False)

    # 5) Summary
    print(f"✓ New CSV saved: {out_csv}")
    print(f"Total {len(corrected)} rows corrected/filled:")
    for mid, old, new in corrected[:10]:
        print(f"  message_id={mid}: {old} → {new}")
    if len(corrected) > 10:
        print(f"  ... and {len(corrected)-10} more.")

if __name__ == "__main__":
    main()