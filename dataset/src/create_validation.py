import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
OUTPUT_DIR = r"C:\riya\ML_Challenge\dataset\validation"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN_DIR}\\train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print("Total S1:", len(gt))

# ---------------------------------------------------------
# Split S1 entities
# ---------------------------------------------------------

train_ids, val_ids = train_test_split(
    gt["source1_entity_id"],
    test_size=0.10,
    random_state=42
)

train_ids = set(train_ids)
val_ids = set(val_ids)

print("Development S1:", len(train_ids))
print("Validation S1:", len(val_ids))

# ---------------------------------------------------------
# Save IDs
# ---------------------------------------------------------

pd.Series(
    list(train_ids),
    name="source1_entity_id"
).to_csv(
    f"{OUTPUT_DIR}\\development_s1_ids.txt",
    index=False
)

pd.Series(
    list(val_ids),
    name="source1_entity_id"
).to_csv(
    f"{OUTPUT_DIR}\\validation_s1_ids.txt",
    index=False
)

print("\nValidation split created.")