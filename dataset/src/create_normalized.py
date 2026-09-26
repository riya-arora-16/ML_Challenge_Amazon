import pandas as pd
import re
import unicodedata
import os

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
OUTPUT_DIR = r"C:\riya\ML_Challenge\dataset\normalized"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    # Unicode normalization
    x = unicodedata.normalize("NFKC", x)

    # Replace punctuation with spaces
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)

    # Normalize whitespace
    x = re.sub(r"\s+", " ", x)

    return x.strip()


def process_file(filename):

    print(f"\nProcessing {filename}...")

    path = f"{TRAIN_DIR}\\{filename}"

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )

    print("Rows:", len(df))

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    output_path = f"{OUTPUT_DIR}\\{filename}"

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print("Saved:", output_path)


process_file("train_source1.tsv")
process_file("train_source2.tsv")
process_file("train_source3.tsv")

print("\nDone.")