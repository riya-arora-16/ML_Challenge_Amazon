import pandas as pd
import re
import random
import os

# ============================================================
# CONFIG
# ============================================================

NORMALIZED_DIR = r"C:\riya\ML_Challenge\dataset\normalized"
TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
VALIDATION_DIR = r"C:\riya\ML_Challenge\dataset\validation"

OUTPUT_FILE = (
    r"C:\riya\ML_Challenge\dataset\validation"
    r"\training_pairs.tsv"
)

S1_SAMPLE_SIZE = 20000
NEGATIVES_PER_S1 = 20

CHUNK_SIZE = 100000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# BLOCKING FUNCTIONS
# ============================================================

def numbers(x):
    if not isinstance(x, str):
        return []

    return re.findall(r"\b\d+[a-z]?\b", x.lower())


def tokens(x):
    if not isinstance(x, str):
        return []

    return re.findall(r"[a-z]+", x.lower())


def name_prefix6(x):
    return x.replace(" ", "")[:6] if x else ""


def name_first_last(x):
    p = x.split() if x else []

    if len(p) == 0:
        return ""

    if len(p) == 1:
        return p[0]

    return p[0] + "|" + p[-1]


def name_first(x):
    p = x.split() if x else []

    return p[0] if p else ""


def address_number(x):
    n = numbers(x)

    return n[0] if n else ""


def address_number_token(x):
    n = numbers(x)
    t = tokens(x)

    if not n or not t:
        return ""

    return n[0] + "|" + t[0]


def address_prefix6(x):
    return x.replace(" ", "")[:6] if x else ""


def addr_number_two_tokens(x):
    n = numbers(x)
    t = tokens(x)

    if not n or len(t) < 2:
        return ""

    return n[0] + "|" + t[0] + "|" + t[1]


def addr_two_numbers(x):
    n = numbers(x)

    if len(n) < 2:
        return ""

    return n[0] + "|" + n[1]


def addr_number_last_token(x):
    n = numbers(x)
    t = tokens(x)

    if not n or not t:
        return ""

    return n[0] + "|" + t[-1]


def addr_first_two_tokens(x):
    return "|".join(tokens(x)[:2])


def addr_first_three_tokens(x):
    return "|".join(tokens(x)[:3])


RULES = {
    "name_prefix6": name_prefix6,
    "name_first_last": name_first_last,
    "name_first": name_first,
    "address_number": address_number,
    "address_number_token": address_number_token,
    "address_prefix6": address_prefix6,
    "addr_number_two_tokens": addr_number_two_tokens,
    "addr_two_numbers": addr_two_numbers,
    "addr_number_last_token": addr_number_last_token,
    "addr_first_two_tokens": addr_first_two_tokens,
    "addr_first_three_tokens": addr_first_three_tokens,
}


# ============================================================
# LOAD S1
# ============================================================

print("=" * 80)
print("RAM-FRIENDLY TRAINING PAIR GENERATION")
print("=" * 80)

print("\nLoading S1...")

s1 = pd.read_csv(
    NORMALIZED_DIR + r"\train_source1.tsv",
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

print("Total S1:", len(s1))

# Random 20k S1
s1 = s1.sample(
    n=S1_SAMPLE_SIZE,
    random_state=RANDOM_SEED
).copy()

print(
    "Sampled S1:",
    len(s1)
)

s1_ids = set(
    s1["entity_id"]
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    TRAIN_DIR + r"\train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

gt = gt[
    gt["source1_entity_id"].isin(s1_ids)
].copy()

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

# True matches for each S1
true_matches = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if row["matched_entity_ids"]:

        true_matches[s1_id] = set(
            row["matched_entity_ids"].split(",")
        )

    else:

        true_matches[s1_id] = set()


total_true = sum(
    len(x)
    for x in true_matches.values()
)

print(
    "True matches belonging to sample:",
    total_true
)


# ============================================================
# CREATE BLOCKING LOOKUPS FROM ONLY 20K S1
# ============================================================

print("\nCreating S1 blocking lookups...")

lookups = {
    rule: {}
    for rule in RULES
}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]

    for rule, func in RULES.items():

        if rule.startswith("name_"):
            value = func(
                row["name_norm"]
            )
        else:
            value = func(
                row["address_norm"]
            )

        if not value:
            continue

        key = (
            country,
            value
        )

        if key not in lookups[rule]:
            lookups[rule][key] = []

        lookups[rule][key].append(
            s1_id
        )


for rule in lookups:

    print(
        rule,
        ":",
        len(lookups[rule]),
        "keys"
    )


# ============================================================
# NEGATIVE SAMPLING STRUCTURES
# ============================================================

# Keep at most NEGATIVES_PER_S1 negatives
negative_samples = {
    s1_id: []
    for s1_id in s1_ids
}

negative_sets = {
    s1_id: set()
    for s1_id in s1_ids
}

candidate_counts = {
    s1_id: 0
    for s1_id in s1_ids
}


# ============================================================
# PROCESS SOURCE CHUNKS
# ============================================================

def process_source(filename):

    print(
        "\nProcessing",
        filename
    )

    path = (
        NORMALIZED_DIR
        + "\\"
        + filename
    )

    processed = 0

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

        processed += len(chunk)

        if chunk_no % 5 == 0:

            print(
                f"  processed {processed:,} rows",
                end="\r"
            )

        # candidate S1 IDs for each row
        row_candidates = {}

        for rule, func in RULES.items():

            if rule.startswith("name_"):

                values = (
                    chunk["name_norm"]
                    .fillna("")
                    .map(func)
                )

            else:

                values = (
                    chunk["address_norm"]
                    .fillna("")
                    .map(func)
                )

            for idx, value in values.items():

                if not value:
                    continue

                country = chunk.at[
                    idx,
                    "country"
                ]

                key = (
                    country,
                    value
                )

                matched_s1 = lookups[
                    rule
                ].get(
                    key
                )

                if not matched_s1:
                    continue

                entity_id = chunk.at[
                    idx,
                    "entity_id"
                ]

                if entity_id not in row_candidates:

                    row_candidates[
                        entity_id
                    ] = set()

                row_candidates[
                    entity_id
                ].update(
                    matched_s1
                )

        # ----------------------------------------------------
        # Add candidates
        # ----------------------------------------------------

        for other_id, s1_candidates in row_candidates.items():

            for s1_id in s1_candidates:

                candidate_counts[
                    s1_id
                ] += 1

                # True positive?
                if (
                    other_id
                    in true_matches.get(
                        s1_id,
                        set()
                    )
                ):
                    continue

                # Already selected?
                if (
                    other_id
                    in negative_sets[s1_id]
                ):
                    continue

                # Reservoir sampling
                current = negative_samples[
                    s1_id
                ]

                if len(current) < NEGATIVES_PER_S1:

                    current.append(
                        other_id
                    )

                    negative_sets[
                        s1_id
                    ].add(
                        other_id
                    )

                else:

                    # Random replacement
                    j = random.randint(
                        0,
                        candidate_counts[
                            s1_id
                        ] - 1
                    )

                    if j < NEGATIVES_PER_S1:

                        old = current[j]

                        negative_sets[
                            s1_id
                        ].remove(old)

                        current[j] = other_id

                        negative_sets[
                            s1_id
                        ].add(other_id)

        del row_candidates
        del chunk

    print(
        f"\nFinished {filename}"
    )


process_source(
    "train_source2.tsv"
)

process_source(
    "train_source3.tsv"
)


# ============================================================
# CREATE FINAL TRAINING PAIRS
# ============================================================

print(
    "\nCreating final training pairs..."
)

rows = []

positive_count = 0
negative_count = 0

for s1_id in s1_ids:

    positives = true_matches.get(
        s1_id,
        set()
    )

    # Positive examples
    for other_id in positives:

        rows.append({
            "source1_entity_id":
                s1_id,

            "other_entity_id":
                other_id,

            "label":
                1
        })

        positive_count += 1

    # Negative examples
    for other_id in negative_samples[
        s1_id
    ]:

        rows.append({
            "source1_entity_id":
                s1_id,

            "other_entity_id":
                other_id,

            "label":
                0
        })

        negative_count += 1


training_pairs = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)

training_pairs.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print(
    "\n" + "=" * 80
)

print(
    "TRAINING PAIRS CREATED"
)

print(
    "=" * 80
)

print(
    "Positive pairs:",
    positive_count
)

print(
    "Negative pairs:",
    negative_count
)

print(
    "Total pairs:",
    len(training_pairs)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "\nCOMPLETE"
)