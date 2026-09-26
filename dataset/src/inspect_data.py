import pandas as pd
import os

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    path = os.path.join(TRAIN_DIR, file)

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nData types:")
    print(df.dtypes)