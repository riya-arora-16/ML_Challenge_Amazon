import pandas as pd
import re
from rapidfuzz.fuzz import ratio

NORMALIZED_DIR = r"C:\riya\ML_Challenge\dataset\normalized"
TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
VALIDATION_FILE = r"C:\riya\ML_Challenge\dataset\validation\validation_s1_ids.txt"

SAMPLE_SIZE = 5000
CHUNK_SIZE = 100000


# =========================================================
# Helpers
# =========================================================

def first_token(x):
    if not isinstance(x, str) or not x:
        return ""
    parts = x.split()
    return parts[0] if parts else ""


def last_token(x):
    if not isinstance(x, str) or not x:
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

    m = re.search(r"\b\d+[a-z]?\b", x)

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


def address_prefix6(x):
    if not isinstance(x, str):
        return ""

    return x.replace(" ", "")[:6]


def similarity(a, b):
    if not isinstance(a, str):
        a = ""

    if not isinstance(b, str):
        b = ""

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


# =========================================================
# 1. Load validation sample
# =========================================================

print("Loading validation IDs...")

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
# 2. Load only the 5K S1 records
# =========================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    f"{NORMALIZED_DIR}\\train_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1 = s1.set_index("entity_id")

print("S1 records:", len(s1))


# =========================================================
# 3. Load ground truth ONLY for those 5K
# =========================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    f"{TRAIN_DIR}\\train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
].copy()

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


# =========================================================
# 4. Build the original blocking keys for S1
# =========================================================

print("\nCreating S1 blocking keys...")

for rule, func, source_col in [
    ("name_prefix6", name_prefix6, "name_norm"),
    ("name_first_last", name_first_last, "name_norm"),
    ("name_first", first_token, "name_norm"),
    ("address_number", address_number, "address_norm"),
    ("address_number_token", address_number_token, "address_norm"),
    ("address_prefix6", address_prefix6, "address_norm"),
]:
    s1[rule] = s1[source_col].map(func)


RULES = [
    "name_prefix6",
    "name_first_last",
    "name_first",
    "address_number",
    "address_number_token",
    "address_prefix6"
]


# =========================================================
# 5. Recreate candidate sets for the 5K S1s
# =========================================================

print("\nBuilding candidate indexes...")

indexes = {
    rule: {}
    for rule in RULES
}


# Only store keys actually needed by our 5K S1 records.
relevant_keys = {}

for rule in RULES:

    relevant_keys[rule] = set(
        (
            row["country"],
            row[rule]
        )
        for _, row in s1.iterrows()
        if row[rule]
    )

    print(
        rule,
        "relevant keys:",
        len(relevant_keys[rule])
    )


# =========================================================
# 6. Process S2/S3 in chunks
# =========================================================

def process_source(filename):

    print("\nProcessing:", filename)

    path = f"{NORMALIZED_DIR}\\{filename}"

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "country",
                "name_norm",
                "address_norm"
            ],
            chunksize=CHUNK_SIZE
        )
    ):

        print(
            f"  Processing chunk {chunk_no + 1}",
            end="\r"
        )

        # Create keys
        chunk["name_prefix6"] = (
            chunk["name_norm"]
            .fillna("")
            .map(name_prefix6)
        )

        chunk["name_first_last"] = (
            chunk["name_norm"]
            .fillna("")
            .map(name_first_last)
        )

        chunk["name_first"] = (
            chunk["name_norm"]
            .fillna("")
            .map(first_token)
        )

        chunk["address_number"] = (
            chunk["address_norm"]
            .fillna("")
            .map(address_number)
        )

        chunk["address_number_token"] = (
            chunk["address_norm"]
            .fillna("")
            .map(address_number_token)
        )

        chunk["address_prefix6"] = (
            chunk["address_norm"]
            .fillna("")
            .map(address_prefix6)
        )

        # Add IDs to indexes only when key is relevant.
        for rule in RULES:

            relevant = relevant_keys[rule]

            for country, key, entity_id in zip(
                chunk["country"],
                chunk[rule],
                chunk["entity_id"]
            ):

                if not key:
                    continue

                lookup = (country, key)

                if lookup not in relevant:
                    continue

                if lookup not in indexes[rule]:
                    indexes[rule][lookup] = []

                indexes[rule][lookup].append(entity_id)

        del chunk

    print()


process_source("train_source2.tsv")
process_source("train_source3.tsv")

print("\nCandidate indexes built.")


# =========================================================
# 7. Find which TRUE matches were missed
# =========================================================

print("\nFinding missed true matches...")

missed_pairs = []

total_true = 0
retrieved_true = 0

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    true_ids = (
        set(row["matched_entity_ids"].split(","))
        if row["matched_entity_ids"]
        else set()
    )

    total_true += len(true_ids)

    s1row = s1.loc[s1_id]

    candidates = set()

    for rule in RULES:

        lookup = (
            s1row["country"],
            s1row[rule]
        )

        candidates.update(
            indexes[rule].get(
                lookup,
                []
            )
        )

    found = true_ids & candidates

    retrieved_true += len(found)

    missing = true_ids - candidates

    for target_id in missing:

        missed_pairs.append({
            "s1_id": s1_id,
            "other_id": target_id
        })


print("\n" + "=" * 80)
print("BLOCKING CHECK")
print("=" * 80)

print(
    "Total true matches:",
    total_true
)

print(
    "Retrieved:",
    retrieved_true
)

print(
    "Missed:",
    len(missed_pairs)
)

print(
    "Recall:",
    f"{retrieved_true / total_true:.2%}"
)


# =========================================================
# 8. IMPORTANT:
#    Now retrieve ONLY the missed IDs from S2/S3
# =========================================================

missed_ids = set(
    x["other_id"]
    for x in missed_pairs
)

print(
    "\nUnique missed entity IDs:",
    len(missed_ids)
)


# =========================================================
# 9. Scan S2/S3 again, but keep ONLY missed IDs
# =========================================================

print("\nRetrieving details of missed records...")

missed_records = {}


def retrieve_missed(filename):

    path = f"{NORMALIZED_DIR}\\{filename}"

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
                "name_norm",
                "address_norm"
            ],
            chunksize=CHUNK_SIZE
        )
    ):

        found = chunk[
            chunk["entity_id"].isin(missed_ids)
        ]

        for _, row in found.iterrows():

            missed_records[
                row["entity_id"]
            ] = row.to_dict()

        del chunk


retrieve_missed("train_source2.tsv")
retrieve_missed("train_source3.tsv")

print(
    "Retrieved missed records:",
    len(missed_records)
)


# =========================================================
# 10. Calculate similarities
# =========================================================

results = []

for pair in missed_pairs:

    s1_id = pair["s1_id"]
    other_id = pair["other_id"]

    if other_id not in missed_records:
        continue

    a = s1.loc[s1_id]
    b = missed_records[other_id]

    results.append({

        "s1_id": s1_id,
        "other_id": other_id,

        "s1_name": a["business_name"],
        "other_name": b["business_name"],

        "s1_address": a["business_address"],
        "other_address": b["business_address"],

        "name_similarity": similarity(
            a["name_norm"],
            b["name_norm"]
        ),

        "address_similarity": similarity(
            a["address_norm"],
            b["address_norm"]
        ),

        "country": a["country"]
    })


result_df = pd.DataFrame(results)


# =========================================================
# 11. Analyse
# =========================================================

print("\n" + "=" * 80)
print("MISSED TRUE MATCH ANALYSIS")
print("=" * 80)

print(
    "Missed pairs:",
    len(result_df)
)

if len(result_df) > 0:

    print("\nSimilarity statistics:")

    print(
        result_df[
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

    print("\nName similarity coverage:")

    for threshold in [
        .3, .4, .5, .6, .7, .8, .9
    ]:

        pct = (
            result_df["name_similarity"]
            >= threshold
        ).mean()

        print(
            f">= {threshold:.1f}: {pct:.2%}"
        )

    print("\nAddress similarity coverage:")

    for threshold in [
        .3, .4, .5, .6, .7, .8, .9
    ]:

        pct = (
            result_df["address_similarity"]
            >= threshold
        ).mean()

        print(
            f">= {threshold:.1f}: {pct:.2%}"
        )

    print("\n" + "=" * 80)
    print("30 LOWEST NAME-SIMILARITY MISSED MATCHES")
    print("=" * 80)

    print(
        result_df
        .sort_values("name_similarity")
        [
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
        .head(30)
        .to_string(index=False)
    )

    output_path = (
        r"C:\riya\ML_Challenge"
        r"\dataset\validation"
        r"\missed_matches.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nDetailed results saved to:",
        output_path
    )

print("\nDone.")