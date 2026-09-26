import pandas as pd
import numpy as np
from difflib import SequenceMatcher

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading files...")

s1 = pd.read_csv(
    f"{TRAIN_DIR}\\train_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "business_address"]
)

s2 = pd.read_csv(
    f"{TRAIN_DIR}\\train_source2.tsv",
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "business_address"]
)

s3 = pd.read_csv(
    f"{TRAIN_DIR}\\train_source3.tsv",
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "business_address"]
)

gt = pd.read_csv(
    f"{TRAIN_DIR}\\train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print("Files loaded.")

# ---------------------------------------------------------
# Prepare ground truth pairs
# ---------------------------------------------------------

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

matches = gt[
    gt["matched_entity_ids"] != ""
][["source1_entity_id", "matched_entity_ids"]].copy()

matches["matched_entity_ids"] = matches["matched_entity_ids"].str.split(",")

matches = matches.explode("matched_entity_ids")

matches = matches.rename(columns={
    "source1_entity_id": "s1_id",
    "matched_entity_ids": "matched_id"
})

# Sample pairs
if len(matches) > SAMPLE_SIZE:
    matches = matches.sample(
        SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

print(f"Analyzing {len(matches):,} true matches.")

# ---------------------------------------------------------
# Prepare source tables
# ---------------------------------------------------------

s1 = s1.rename(columns={
    "entity_id": "s1_id",
    "business_name": "name_s1",
    "business_address": "address_s1"
})

others = pd.concat([s2, s3], ignore_index=True)

others = others.rename(columns={
    "entity_id": "matched_id",
    "business_name": "name_other",
    "business_address": "address_other"
})

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

matches = matches.merge(
    s1,
    on="s1_id",
    how="left"
)

matches = matches.merge(
    others,
    on="matched_id",
    how="left"
)

print("Pairs merged.")

# ---------------------------------------------------------
# Cleaning for similarity
# ---------------------------------------------------------

def clean_text(x):
    if pd.isna(x):
        return ""

    return str(x).lower().strip()


def char_similarity(a, b):
    return SequenceMatcher(
        None,
        clean_text(a),
        clean_text(b)
    ).ratio()


def token_similarity(a, b):
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union else 0.0


# ---------------------------------------------------------
# Calculate similarities
# ---------------------------------------------------------

print("Calculating similarities...")

matches["name_char_sim"] = [
    char_similarity(a, b)
    for a, b in zip(
        matches["name_s1"],
        matches["name_other"]
    )
]

matches["address_char_sim"] = [
    char_similarity(a, b)
    for a, b in zip(
        matches["address_s1"],
        matches["address_other"]
    )
]

matches["name_token_sim"] = [
    token_similarity(a, b)
    for a, b in zip(
        matches["name_s1"],
        matches["name_other"]
    )
]

matches["address_token_sim"] = [
    token_similarity(a, b)
    for a, b in zip(
        matches["address_s1"],
        matches["address_other"]
    )
]

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRUE MATCH SIMILARITY")
print("=" * 70)

columns = [
    "name_char_sim",
    "name_token_sim",
    "address_char_sim",
    "address_token_sim"
]

print(
    matches[columns].describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

# ---------------------------------------------------------
# Useful threshold recall
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NAME CHARACTER SIMILARITY COVERAGE")
print("=" * 70)

for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:

    recall = (
        matches["name_char_sim"] >= threshold
    ).mean()

    print(
        f"Name similarity >= {threshold:.2f}: "
        f"{recall:.2%}"
    )

print("\n" + "=" * 70)
print("ADDRESS CHARACTER SIMILARITY COVERAGE")
print("=" * 70)

for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:

    recall = (
        matches["address_char_sim"] >= threshold
    ).mean()

    print(
        f"Address similarity >= {threshold:.2f}: "
        f"{recall:.2%}"
    )

# ---------------------------------------------------------
# Show difficult examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOW NAME SIMILARITY TRUE MATCHES")
print("=" * 70)

low_name = matches.sort_values(
    "name_char_sim"
).head(20)

print(
    low_name[
        [
            "name_s1",
            "name_other",
            "name_char_sim",
            "address_s1",
            "address_other"
        ]
    ].to_string(index=False)
)