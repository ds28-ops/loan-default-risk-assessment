import pandas as pd
import os

chunks = pd.read_csv("/mnt/data/raw/accepted_2007_to_2018Q4.csv", chunksize=10000)

drop_cols = ["id", "member_id", "url", "desc", "title"]

def map_status(status):
    if status in ["Fully Paid", "Current"]:
        return "High"
    elif status in ["Late (31-120 days)", "In Grace Period", "Late (16-30 days)"]:
        return "Medium"
    else:
        return "Low"

output_path = "/mnt/data/processed/cleaned_data.csv"
first_chunk = True

for chunk in chunks:
    chunk.drop(columns=[col for col in drop_cols if col in chunk.columns], inplace=True, errors="ignore")
    chunk["risk_level"] = chunk["loan_status"].apply(map_status)
    chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
    first_chunk = False

print("✅ Done")
