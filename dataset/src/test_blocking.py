import pandas as pd
import re
from collections import defaultdict, Counter

TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
NORMALIZED_DIR = r"C:\riya\ML_Challenge\dataset\normalized"
VALIDATION_FILE = r"C:\riya\ML_Challenge\dataset\validation\validation_s1_ids.txt"

SAMPLE_SIZE = 5000
CHUNK_SIZE = 200000


# =========================================================
# KEY FUNCTIONS
# =========================================================

def tokens(text):
    if not isinstance(text, str) or not text:
        return []
    return text.split()


def first_token(text):
    t = tokens(text)
    return t[0] if t else ""


def last_token(text):
    t = tokens(text)
    return t[-1] if t else ""


def name_prefix6(text):
    if not isinstance(text, str):
        return ""
    return text.replace(" ", "")[:6]


def name_first_last(text):
    t = tokens(text)

    if len(t) < 2:
        return t[0] if t else ""

    return t[0] + "|" + t[-1]


def address_number(text):
    if not isinstance(text, str):
        return ""

    # Find first number / house number
    m = re.search(r"\b\d+[a-z]?\b", text)

    return m.group(0) if m else ""


def address_number_token(text):
    if not isinstance(text, str):
        return ""

    number = address_number(text)

    if not number:
        return ""

    # Remove the number and find first alphabetic token
    remaining = re.sub(
        r"\b\d+[a-z]?\b",
        " ",
        text
    )

    alpha_tokens = re.findall(
        r"[a-z]+",
        remaining
    )

    if not alpha_tokens:
        return number

    return number + "|" + alpha_tokens[0]


def address_prefix6(text):
    if not isinstance(text, str):
        return ""

    return text.replace(" ", "")[:6]


# =========================================================
# LOAD VALIDATION S1
# =========================================================

print("Loading validation IDs...")

val_ids = pd.read_csv(
    VALIDATION_FILE,
    dtype=str
)["source1_entity_id"]

val_ids = val_ids.sample(
    n=SAMPLE_SIZE,
    random_state=42
)

val_ids = set(val_ids)

print("Validation sample:", len(val_ids))


# =========================================================
# LOAD S1
# =========================================================

print("\nLoading Source 1...")

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

print("S1 records:", len(s1))


# =========================================================
# CREATE S1 KEYS
# =========================================================

print("\nCreating S1 blocking keys...")

s1["name_prefix6"] = s1["name_norm"].map(name_prefix6)

s1["name_first_last"] = (
    s1["name_norm"].map(name_first_last)
)

s1["name_first"] = (
    s1["name_norm"].map(first_token)
)

s1["name_last"] = (
    s1["name_norm"].map(last_token)
)

s1["address_number"] = (
    s1["address_norm"].map(address_number)
)

s1["address_number_token"] = (
    s1["address_norm"].map(address_number_token)
)

s1["address_prefix6"] = (
    s1["address_norm"].map(address_prefix6)
)


# =========================================================
# BLOCKING RULES
# =========================================================

RULES = [
    "name_prefix6",
    "name_first_last",
    "name_first",
    "address_number",
    "address_number_token",
    "address_prefix6"
]


# =========================================================
# CREATE RELEVANT KEY SETS
#
# Only keys occurring in our 5000 S1 records are indexed.
# This dramatically reduces memory.
# =========================================================

relevant_keys = {}

for rule in RULES:

    relevant_keys[rule] = set()

    for country, key in zip(
        s1["country"],
        s1[rule]
    ):

        if key:
            relevant_keys[rule].add(
                (country, key)
            )

    print(
        rule,
        "relevant keys:",
        len(relevant_keys[rule])
    )


# =========================================================
# BUILD CANDIDATE INDEXES
# =========================================================

indexes = {
    rule: defaultdict(set)
    for rule in RULES
}


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
            f"  chunk {chunk_no + 1}",
            end="\r"
        )

        # ---------------------------------------------
        # Generate keys
        # ---------------------------------------------

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

        chunk["name_last"] = (
            chunk["name_norm"]
            .fillna("")
            .map(last_token)
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

        # ---------------------------------------------
        # Insert only relevant keys
        # ---------------------------------------------

        for rule in RULES:

            relevant = relevant_keys[rule]

            for country, key, entity_id in zip(
                chunk["country"],
                chunk[rule],
                chunk["entity_id"]
            ):

                if not key:
                    continue

                lookup_key = (country, key)

                if lookup_key in relevant:

                    indexes[rule][lookup_key].add(
                        entity_id
                    )

    print()


process_source("train_source2.tsv")
process_source("train_source3.tsv")

print("\nBlocking indexes built.")


# =========================================================
# LOAD GROUND TRUTH
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
# CREATE TRUE MATCH DICTIONARY
# =========================================================

true_matches = {}

for _, row in gt.iterrows():

    ids = row["matched_entity_ids"]

    if ids:
        true_matches[
            row["source1_entity_id"]
        ] = set(ids.split(","))
    else:
        true_matches[
            row["source1_entity_id"]
        ] = set()


# =========================================================
# EVALUATE EACH BLOCKING RULE
# =========================================================

print("\n" + "=" * 80)
print("INDIVIDUAL BLOCKING RULE PERFORMANCE")
print("=" * 80)

rule_results = {}

for rule in RULES:

    total_true = 0
    retrieved = 0
    total_candidates = 0

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        true = true_matches.get(
            s1_id,
            set()
        )

        key = (
            row["country"],
            row[rule]
        )

        candidates = indexes[rule].get(
            key,
            set()
        )

        total_true += len(true)

        retrieved += len(
            true & candidates
        )

        total_candidates += len(candidates)

    recall = (
        retrieved / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates / len(s1)
    )

    rule_results[rule] = {
        "recall": recall,
        "avg_candidates": avg_candidates
    }

    print(
        f"\n{rule}"
    )

    print(
        f"  Recall: "
        f"{recall:.2%}"
    )

    print(
        f"  Avg candidates: "
        f"{avg_candidates:,.1f}"
    )


# =========================================================
# COMBINED BLOCKING
# =========================================================

print("\n" + "=" * 80)
print("COMBINED BLOCKING")
print("=" * 80)

total_true = 0
retrieved = 0
total_candidates = 0

for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    true = true_matches.get(
        s1_id,
        set()
    )

    candidates = set()

    # Union of all blocking rules
    for rule in RULES:

        key = (
            row["country"],
            row[rule]
        )

        candidates.update(
            indexes[rule].get(
                key,
                set()
            )
        )

    total_true += len(true)

    retrieved += len(
        true & candidates
    )

    total_candidates += len(candidates)


combined_recall = (
    retrieved / total_true
    if total_true
    else 0
)

avg_candidates = (
    total_candidates / len(s1)
)

print(
    f"\nTrue matches: "
    f"{total_true:,}"
)

print(
    f"Retrieved true matches: "
    f"{retrieved:,}"
)

print(
    f"Combined blocking recall: "
    f"{combined_recall:.2%}"
)

print(
    f"Average candidates/S1: "
    f"{avg_candidates:,.1f}"
)


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 80)
print("BLOCKING EXPERIMENT COMPLETE")
print("=" * 80)