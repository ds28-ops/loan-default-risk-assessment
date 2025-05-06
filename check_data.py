import pandas as pd

file_path = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
df = pd.read_csv(file_path)

# Display column names for inspection
print(df.columns.tolist())