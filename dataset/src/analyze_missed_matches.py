import pandas as pd
import re
from rapidfuzz.fuzz import ratio, token_set_ratio

NORMALIZED_DIR = r"C:\riya\ML_Challenge\dataset\normalized"
TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
VALIDATION_FILE = r"C:\riya\ML_Challenge\dataset\validation\validation_s1_ids.txt"

SAMPLE_SIZE = 5000


# =========================================================
# Helpers
# =========================================================

def normalize_id_list(x):
    if pd.isna(x) or not str(x).strip():
        return set()

    return set(str(x).split(","))


def sim(a, b):
    if not isinstance(a, str):
        a = ""

    if not isinstance(b, str):
        b = ""

    if not a or not b:
        return 0

    return ratio(a, b) / 100


# =========================================================
# Validation sample
# =========================================================

print("Loading validation sample...")

val_ids = pd.read_csv(
    VALIDATION_FILE,
    dtype=str
)["source1_entity_id"]

val_ids = set(
    val_ids.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )
)

print("Validation S1:", len(val_ids))


# =========================================================
# Load S1
# =========================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    f"{NORMALIZED_DIR}\\train_source1.tsv",
    sep="\t",
    dtype=str
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1 = s1.set_index("entity_id")


# =========================================================
# Load S2/S3 records
# =========================================================

print("Loading S2...")

s2 = pd.read_csv(
    f"{NORMALIZED_DIR}\\train_source2.tsv",
    sep="\t",
    dtype=str
)

print("Loading S3...")

s3 = pd.read_csv(
    f"{NORMALIZED_DIR}\\train_source3.tsv",
    sep="\t",
    dtype=str
)

other = pd.concat(
    [s2, s3],
    ignore_index=True
)

other = other.set_index("entity_id")


# =========================================================
# Ground truth
# =========================================================

print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN_DIR}\\train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
].copy()


# =========================================================
# Recreate V2 blocking rules
# =========================================================

def first_token(x):
    if not isinstance(x, str):
        return ""
    parts = x.split()
    return parts[0] if parts else ""


def last_token(x):
    if not isinstance(x, str):
        return ""
    parts = x.split()
    return parts[-1] if parts else ""


def name_prefix6(x):
    if not isinstance(x, str):
        return ""
    return x.replace(" ", "")[:6]


def name_first_last(x):
    if not isinstance(x, str):
        return ""

    parts = x.split()

    if len(parts) == 0:
        return ""

    if len(parts) == 1:
        return parts[0]

    return parts[0] + "|" + parts[-1]


def address_number(x):
    if not isinstance(x, str):
        return ""

    m = re.search(
        r"\b\d+[a-z]?\b",
        x
    )

    return m.group(0) if m else ""


def address_number_token(x):
    if not isinstance(x, str):
        return ""

    num = address_number(x)

    if not num:
        return ""

    remaining = re.sub(
        r"\b\d+[a-z]?\b",
        " ",
        x
    )

    words = re.findall(
        r"[a-z]+",
        remaining
    )

    if not words:
        return num

    return num + "|" + words[0]


# =========================================================
# Build indexes
# =========================================================

RULES = [
    "name_prefix6",
    "name_first_last",
    "name_first",
    "address_number",
    "address_number_token",
    "address_prefix6"
]


def add_keys(df):

    df = df.copy()

    df["name_prefix6"] = (
        df["name_norm"].map(name_prefix6)
    )

    df["name_first_last"] = (
        df["name_norm"].map(name_first_last)
    )

    df["name_first"] = (
        df["name_norm"].map(first_token)
    )

    df["address_number"] = (
        df["address_norm"].map(address_number)
    )

    df["address_number_token"] = (
        df["address_norm"].map(address_number_token)
    )

    df["address_prefix6"] = (
        df["address_norm"]
        .fillna("")
        .str.replace(" ", "", regex=False)
        .str[:6]
    )

    return df


print("\nCreating keys...")

s1 = add_keys(s1)
other = add_keys(other.reset_index())
other = other.set_index("entity_id")


indexes = {
    rule: {}
    for rule in RULES
}


for rule in RULES:

    print("Building:", rule)

    temp = {}

    for entity_id, row in other.iterrows():

        key = (
            row["country"],
            row[rule]
        )

        if not row[rule]:
            continue

        if key not in temp:
            temp[key] = []

        temp[key].append(entity_id)

    indexes[rule] = temp


# =========================================================
# Find missed matches
# =========================================================

print("\nFinding missed true matches...")

missed = []

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if not row["matched_entity_ids"] or pd.isna(
        row["matched_entity_ids"]
    ):
        continue

    s1row = s1.loc[s1_id]

    candidates = set()

    for rule in RULES:

        key = (
            s1row["country"],
            s1row[rule]
        )

        candidates.update(
            indexes[rule].get(
                key,
                []
            )
        )

    true_ids = normalize_id_list(
        row["matched_entity_ids"]
    )

    missing = true_ids - candidates

    for target_id in missing:

        if target_id not in other.index:
            continue

        target = other.loc[target_id]

        missed.append({

            "s1_id": s1_id,
            "other_id": target_id,

            "s1_name": s1row["business_name"],
            "other_name": target["business_name"],

            "s1_name_norm": s1row["name_norm"],
            "other_name_norm": target["name_norm"],

            "s1_address": s1row["business_address"],
            "other_address": target["business_address"],

            "name_similarity": sim(
                s1row["name_norm"],
                target["name_norm"]
            ),

            "address_similarity": sim(
                s1row["address_norm"],
                target["address_norm"]
            ),

            "country": s1row["country"]
        })


# =========================================================
# Results
# =========================================================

missed_df = pd.DataFrame(missed)

print("\n" + "=" * 80)
print("MISSED TRUE MATCH ANALYSIS")
print("=" * 80)

print(
    "Missed true pairs:",
    len(missed_df)
)

if len(missed_df) > 0:

    print("\nSimilarity statistics:")

    print(
        missed_df[
            [
                "name_similarity",
                "address_similarity"
            ]
        ].describe(
            percentiles=[
                .10,
                .25,
                .50,
                .75,
                .90
            ]
        )
    )

    print("\nName similarity >= threshold:")

    for threshold in [
        .3,
        .4,
        .5,
        .6,
        .7,
        .8,
        .9
    ]:

        pct = (
            missed_df["name_similarity"]
            >= threshold
        ).mean()

        print(
            f">= {threshold:.1f}: {pct:.2%}"
        )

    print("\nAddress similarity >= threshold:")

    for threshold in [
        .3,
        .4,
        .5,
        .6,
        .7,
        .8,
        .9
    ]:

        pct = (
            missed_df["address_similarity"]
            >= threshold
        ).mean()

        print(
            f">= {threshold:.1f}: {pct:.2%}"
        )

    print("\nExamples of missed matches:")

    print(
        missed_df[
            [
                "s1_name",
                "other_name",
                "s1_address",
                "other_address",
                "name_similarity",
                "address_similarity",
                "country"
            ]
        ]
        .sort_values(
            "name_similarity"
        )
        .head(30)
        .to_string(index=False)
    )

    output_path = (
        r"C:\riya\ML_Challenge"
        r"\dataset\validation"
        r"\missed_matches.csv"
    )

    missed_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved detailed analysis to:",
        output_path
    )

else:

    print("No missed matches found.")