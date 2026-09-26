import pandas as pd

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"

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
# Prepare ground truth
# ---------------------------------------------------------

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

# Convert comma-separated matches into individual rows
matches = gt[
    gt["matched_entity_ids"] != ""
][["source1_entity_id", "matched_entity_ids"]].copy()

matches["matched_entity_ids"] = matches["matched_entity_ids"].str.split(",")

matches = matches.explode("matched_entity_ids")

matches = matches.rename(columns={
    "source1_entity_id": "s1_id",
    "matched_entity_ids": "matched_id"
})

print("Total ground-truth pairs:", len(matches))

# ---------------------------------------------------------
# Combine S2 + S3
# ---------------------------------------------------------

others = pd.concat(
    [s2, s3],
    ignore_index=True
)

others = others.rename(columns={
    "entity_id": "matched_id",
    "business_name": "name_other",
    "business_address": "address_other"
})

s1 = s1.rename(columns={
    "entity_id": "s1_id",
    "business_name": "name_s1",
    "business_address": "address_s1"
})

# ---------------------------------------------------------
# Merge S1 information onto matches
# ---------------------------------------------------------

matches = matches.merge(
    s1,
    on="s1_id",
    how="left"
)

# ---------------------------------------------------------
# Merge matched S2/S3 information
# ---------------------------------------------------------

matches = matches.merge(
    others,
    on="matched_id",
    how="left"
)

print("Merged successfully.")

# ---------------------------------------------------------
# Exact comparisons
# ---------------------------------------------------------

name_valid = (
    matches["name_s1"].notna()
    & matches["name_other"].notna()
)

address_valid = (
    matches["address_s1"].notna()
    & matches["address_other"].notna()
)

exact_name = (
    name_valid
    & matches["name_s1"].eq(matches["name_other"])
)

exact_address = (
    address_valid
    & matches["address_s1"].eq(matches["address_other"])
)

exact_both = exact_name & exact_address

total = len(matches)

print("\n" + "=" * 70)
print("EXACT MATCH ANALYSIS")
print("=" * 70)

print(f"\nTotal ground-truth matches: {total:,}")

print(
    f"\nExact name matches: {exact_name.sum():,}"
    f" ({exact_name.mean():.2%})"
)

print(
    f"Exact address matches: {exact_address.sum():,}"
    f" ({exact_address.mean():.2%})"
)

print(
    f"Exact name + address: {exact_both.sum():,}"
    f" ({exact_both.mean():.2%})"
)