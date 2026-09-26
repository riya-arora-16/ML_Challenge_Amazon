import pandas as pd
import re

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]

for file in files:

    path = f"{TRAIN_DIR}\\{file}"

    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address",
            "country"
        ]
    )

    # -------------------------------------------------
    # BASIC STATISTICS
    # -------------------------------------------------

    print("\nRows:", len(df))

    print("\nCountries:")
    print(df["country"].value_counts(dropna=False).head(20))

    print("\nMissing names:", df["business_name"].isna().sum())
    print("Missing addresses:", df["business_address"].isna().sum())

    # -------------------------------------------------
    # NAME LENGTH
    # -------------------------------------------------

    name_lengths = df["business_name"].fillna("").str.len()

    print("\nName length:")
    print(name_lengths.describe())

    # -------------------------------------------------
    # ADDRESS LENGTH
    # -------------------------------------------------

    address_lengths = df["business_address"].fillna("").str.len()

    print("\nAddress length:")
    print(address_lengths.describe())

    # -------------------------------------------------
    # NON-ASCII NAMES
    # -------------------------------------------------

    non_ascii_name = (
        df["business_name"]
        .fillna("")
        .str.contains(r"[^\x00-\x7F]", regex=True)
    )

    print("\nNames containing non-ASCII characters:",
          non_ascii_name.sum())

    print("\nSample non-ASCII names:")

    print(
        df.loc[non_ascii_name, "business_name"]
        .drop_duplicates()
        .head(15)
        .to_string(index=False)
    )

    # -------------------------------------------------
    # NAMES CONTAINING NUMBERS
    # -------------------------------------------------

    numeric_name = (
        df["business_name"]
        .fillna("")
        .str.contains(r"\d", regex=True)
    )

    print("\nNames containing numbers:",
          numeric_name.sum())

    # -------------------------------------------------
    # URL-LIKE BUSINESS NAMES
    # -------------------------------------------------

    url_name = (
        df["business_name"]
        .fillna("")
        .str.contains(
            r"(www\.|\.com\b|\.in\b|\.net\b|\.org\b)",
            regex=True,
            case=False
        )
    )

    print("\nURL-like names:", url_name.sum())

    print("\nSample URL-like names:")

    print(
        df.loc[url_name, "business_name"]
        .drop_duplicates()
        .head(15)
        .to_string(index=False)
    )

    # -------------------------------------------------
    # COMMON LEGAL TERMS
    # -------------------------------------------------

    legal_terms = [
        "ltd",
        "limited",
        "llc",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "pvt",
        "private",
        "llp",
        "plc"
    ]

    print("\nLegal term frequencies:")

    names = df["business_name"].fillna("").str.lower()

    for term in legal_terms:

        count = names.str.contains(
            rf"\b{re.escape(term)}\b",
            regex=True
        ).sum()

        if count > 0:
            print(f"{term:15} {count}")