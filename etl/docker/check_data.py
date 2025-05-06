import pandas as pd

file_path = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ Loaded CSV successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 50 columns:")
    print(df.columns[:50].tolist())
    print("\nRemaining columns:")
    print(df.columns[50:].tolist())
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
