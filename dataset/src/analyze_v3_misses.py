import pandas as pd
import re
from rapidfuzz.fuzz import ratio

# =========================================================
# CONFIG
# =========================================================

NORMALIZED_DIR = r"C:\riya\ML_Challenge\dataset\normalized"
TRAIN_DIR = r"C:\riya\ML_Challenge\dataset\train"
VALIDATION_FILE = (
    r"C:\riya\ML_Challenge\dataset\validation"
    r"\validation_s1_ids.txt"
)

OUTPUT_FILE = (
    r"C:\riya\ML_Challenge\dataset\validation"
    r"\v3_missed_matches.tsv"
)

SAMPLE_SIZE = 5000
CHUNK_SIZE = 100000


# =========================================================
# HELPERS
# =========================================================

def get_numbers(text):
    if not isinstance(text, str):
        return []

    return re.findall(
        r"\b\d+[a-z]?\b",
        text.lower()
    )


def get_tokens(text):
    if not isinstance(text, str):
        return []

    return re.findall(
        r"[a-z]+",
        text.lower()
    )


def token_set(text):
    return set(get_tokens(text))


def number_overlap(a, b):

    a = set(get_numbers(a))
    b = set(get_numbers(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def token_overlap(a, b):

    a = token_set(a)
    b = token_set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def common_tokens(a, b):

    return sorted(
        token_set(a) & token_set(b)
    )


# =========================================================
# LOAD VALIDATION IDS
# =========================================================

print("=" * 80)
print("RAM-FRIENDLY V3 MISSED MATCH ANALYSIS")
print("=" * 80)

print("\nLoading validation IDs...")

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

print(
    "Validation S1:",
    len(val_ids)
)


# =========================================================
# LOAD S1 ONLY
# =========================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    f"{NORMALIZED_DIR}\\train_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "country",
        "name_norm",
        "address_norm"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1 = s1.set_index(
    "entity_id"
)

print(
    "S1 records:",
    len(s1)
)


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
# V3 BLOCKING FUNCTIONS
# =========================================================

def name_prefix6(x):

    if not isinstance(x, str):
        return ""

    return x.replace(" ", "")[:6]


def name_first(x):

    if not isinstance(x, str):
        return ""

    p = x.split()

    return p[0] if p else ""


def name_first_last(x):

    if not isinstance(x, str):
        return ""

    p = x.split()

    if not p:
        return ""

    if len(p) == 1:
        return p[0]

    return p[0] + "|" + p[-1]


def address_number(x):

    nums = get_numbers(x)

    return nums[0] if nums else ""


def address_number_token(x):

    nums = get_numbers(x)
    toks = get_tokens(x)

    if not nums or not toks:
        return ""

    return nums[0] + "|" + toks[0]


def address_prefix6(x):

    if not isinstance(x, str):
        return ""

    return x.replace(" ", "")[:6]


def addr_number_two_tokens(x):

    nums = get_numbers(x)
    toks = get_tokens(x)

    if not nums or len(toks) < 2:
        return ""

    return (
        nums[0]
        + "|"
        + toks[0]
        + "|"
        + toks[1]
    )


def addr_two_numbers(x):

    nums = get_numbers(x)

    if len(nums) < 2:
        return ""

    return nums[0] + "|" + nums[1]


def addr_number_last_token(x):

    nums = get_numbers(x)
    toks = get_tokens(x)

    if not nums or not toks:
        return ""

    return nums[0] + "|" + toks[-1]


def addr_first_two_tokens(x):

    toks = get_tokens(x)

    return "|".join(toks[:2])


def addr_first_three_tokens(x):

    toks = get_tokens(x)

    return "|".join(toks[:3])


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
    "addr_first_three_tokens": addr_first_three_tokens
}


# =========================================================
# FIRST: BUILD S1 KEYS
# =========================================================

print("\nCreating V3 keys for validation S1...")

for rule, func in RULES.items():

    if rule.startswith("name_"):

        s1[rule] = (
            s1["name_norm"]
            .fillna("")
            .map(func)
        )

    else:

        s1[rule] = (
            s1["address_norm"]
            .fillna("")
            .map(func)
        )


# =========================================================
# FIND V3 CANDIDATES FOR ONLY THE 5000 S1s
# =========================================================
#
# IMPORTANT:
# We do NOT build a global S2/S3 index.
#
# Instead we process each source in chunks and only retain
# candidates that belong to our 5000 validation records.
#
# =========================================================

print("\nFinding V3 candidates using chunks...")

candidate_map = {
    s1_id: set()
    for s1_id in s1.index
}


def process_source(filename):

    print(
        f"\nScanning {filename}..."
    )

    path = (
        f"{NORMALIZED_DIR}\\{filename}"
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

        if chunk_no % 10 == 0:
            print(
                f"  processed "
                f"{processed:,} rows",
                end="\r"
            )

        # ---------------------------------------------
        # For each rule, create a lookup from
        # (country, key) -> validation S1 IDs
        # ---------------------------------------------

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

            # Map validation S1 key -> S1 IDs
            lookup = {}

            for s1_id in s1.index:

                key = s1.loc[
                    s1_id,
                    rule
                ]

                if not key:
                    continue

                compound = (
                    s1.loc[
                        s1_id,
                        "country"
                    ],
                    key
                )

                lookup.setdefault(
                    compound,
                    []
                ).append(
                    s1_id
                )

            # Process only non-empty values
            for country, key, entity_id in zip(
                chunk["country"],
                values,
                chunk["entity_id"]
            ):

                if not key:
                    continue

                compound = (
                    country,
                    key
                )

                matching_s1s = lookup.get(
                    compound
                )

                if matching_s1s:

                    for s1_id in matching_s1s:

                        candidate_map[
                            s1_id
                        ].add(entity_id)

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


# =========================================================
# IDENTIFY MISSED TRUE MATCHES
# =========================================================

print(
    "\nIdentifying V3 missed true matches..."
)

missed_pairs = []

total_true = 0
retrieved = 0

for _, row in gt.iterrows():

    s1_id = row[
        "source1_entity_id"
    ]

    if not row[
        "matched_entity_ids"
    ]:

        continue

    true_ids = set(
        row[
            "matched_entity_ids"
        ].split(",")
    )

    total_true += len(
        true_ids
    )

    candidates = candidate_map.get(
        s1_id,
        set()
    )

    found = (
        true_ids & candidates
    )

    retrieved += len(found)

    missed = (
        true_ids - candidates
    )

    for other_id in missed:

        missed_pairs.append(
            (
                s1_id,
                other_id
            )
        )


print(
    "\nTotal true matches:",
    total_true
)

print(
    "Retrieved:",
    retrieved
)

print(
    "Missed:",
    len(missed_pairs)
)


# =========================================================
# FREE LARGE CANDIDATE STRUCTURES
# =========================================================

del candidate_map
del gt


# =========================================================
# CREATE SETS OF NEEDED S2/S3 IDs
# =========================================================

needed_s2 = set()
needed_s3 = set()

for _, other_id in missed_pairs:

    if other_id.startswith("S2-"):
        needed_s2.add(other_id)

    elif other_id.startswith("S3-"):
        needed_s3.add(other_id)


print(
    "\nMissed S2 records needed:",
    len(needed_s2)
)

print(
    "Missed S3 records needed:",
    len(needed_s3)
)


# =========================================================
# SCAN ONLY FOR THE MISSED RECORDS
# =========================================================

print(
    "\nExtracting missed records from S2/S3..."
)


other_records = {}


def extract_records(
    filename,
    wanted_ids
):

    if not wanted_ids:
        return

    print(
        f"\nScanning {filename}..."
    )

    path = (
        f"{NORMALIZED_DIR}\\{filename}"
    )

    found = 0
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

        mask = chunk[
            "entity_id"
        ].isin(wanted_ids)

        selected = chunk[
            mask
        ]

        for _, row in selected.iterrows():

            other_records[
                row["entity_id"]
            ] = {
                "country":
                    row["country"],

                "name":
                    row["name_norm"],

                "address":
                    row["address_norm"]
            }

            found += 1

        if chunk_no % 10 == 0:

            print(
                f"  scanned "
                f"{processed:,} rows | "
                f"found {found:,}",
                end="\r"
            )

        if found == len(wanted_ids):
            break

        del chunk

    print(
        f"\nFinished. Found {found}"
    )


extract_records(
    "train_source2.tsv",
    needed_s2
)

extract_records(
    "train_source3.tsv",
    needed_s3
)


# =========================================================
# CALCULATE SIMILARITIES
# =========================================================

print(
    "\nCalculating similarities..."
)

results = []


for s1_id, other_id in missed_pairs:

    if s1_id not in s1.index:
        continue

    if other_id not in other_records:
        continue

    s1row = s1.loc[
        s1_id
    ]

    other = other_records[
        other_id
    ]

    name1 = str(
        s1row["name_norm"]
    )

    name2 = str(
        other["name"]
    )

    addr1 = str(
        s1row["address_norm"]
    )

    addr2 = str(
        other["address"]
    )

    name_sim = (
        ratio(
            name1,
            name2
        ) / 100
    )

    address_sim = (
        ratio(
            addr1,
            addr2
        ) / 100
    )

    num_overlap = (
        number_overlap(
            addr1,
            addr2
        )
    )

    tok_overlap = (
        token_overlap(
            addr1,
            addr2
        )
    )

    common = ",".join(
        common_tokens(
            addr1,
            addr2
        )
    )

    results.append({

        "s1_id":
            s1_id,

        "other_id":
            other_id,

        "country":
            s1row["country"],

        "name_s1":
            name1,

        "name_other":
            name2,

        "address_s1":
            addr1,

        "address_other":
            addr2,

        "name_sim":
            name_sim,

        "address_sim":
            address_sim,

        "number_overlap":
            num_overlap,

        "token_overlap":
            tok_overlap,

        "common_address_tokens":
            common
    })


# =========================================================
# RESULTS
# =========================================================

missed_df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 80
)

print(
    "MISSED MATCH STATISTICS"
)

print(
    "=" * 80
)

print(
    "Rows analyzed:",
    len(missed_df)
)

if len(missed_df) > 0:

    print(
        "\nName similarity:"
    )

    print(
        missed_df[
            "name_sim"
        ].describe(
            percentiles=[
                .25,
                .50,
                .75,
                .90
            ]
        )
    )

    print(
        "\nAddress similarity:"
    )

    print(
        missed_df[
            "address_sim"
        ].describe(
            percentiles=[
                .25,
                .50,
                .75,
                .90
            ]
        )
    )

    print(
        "\nNumber overlap:"
    )

    print(
        missed_df[
            "number_overlap"
        ].describe(
            percentiles=[
                .25,
                .50,
                .75,
                .90
            ]
        )
    )

    print(
        "\nAddress token overlap:"
    )

    print(
        missed_df[
            "token_overlap"
        ].describe(
            percentiles=[
                .25,
                .50,
                .75,
                .90
            ]
        )
    )


# =========================================================
# ADDRESS THRESHOLDS
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "MISSED PAIRS BY ADDRESS SIGNAL"
)

print(
    "=" * 80
)

if len(missed_df) > 0:

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]:

        count = (
            missed_df[
                "address_sim"
            ] >= threshold
        ).sum()

        print(
            f"Address similarity >= "
            f"{threshold:.2f}: "
            f"{count:,} "
            f"({count / len(missed_df):.2%})"
        )


# =========================================================
# NUMBER SIGNAL
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "MISSED PAIRS BY NUMBER OVERLAP"
)

print(
    "=" * 80
)

if len(missed_df) > 0:

    for threshold in [
        0.25,
        0.50,
        0.75,
        1.00
    ]:

        count = (
            missed_df[
                "number_overlap"
            ] >= threshold
        ).sum()

        print(
            f"Number overlap >= "
            f"{threshold:.2f}: "
            f"{count:,} "
            f"({count / len(missed_df):.2%})"
        )


# =========================================================
# COUNTRY
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "MISSED MATCHES BY COUNTRY"
)

print(
    "=" * 80
)

if len(missed_df) > 0:

    print(
        missed_df[
            "country"
        ].value_counts()
    )


# =========================================================
# LOW NAME / HIGH ADDRESS
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "LOW NAME + HIGH ADDRESS"
)

print(
    "=" * 80
)

if len(missed_df) > 0:

    interesting = missed_df[
        (
            missed_df["name_sim"] < 0.50
        )
        &
        (
            missed_df["address_sim"] >= 0.70
        )
    ]

    print(
        "Count:",
        len(interesting)
    )

    print(
        interesting[
            [
                "name_s1",
                "name_other",
                "name_sim",
                "address_s1",
                "address_other",
                "address_sim",
                "number_overlap",
                "common_address_tokens"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# =========================================================
# HIGH NUMBER OVERLAP
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "HIGH NUMBER OVERLAP"
)

print(
    "=" * 80
)

if len(missed_df) > 0:

    interesting = missed_df[
        missed_df[
            "number_overlap"
        ] >= 0.50
    ]

    print(
        "Count:",
        len(interesting)
    )

    print(
        interesting[
            [
                "name_s1",
                "name_other",
                "address_s1",
                "address_other",
                "number_overlap",
                "token_overlap",
                "common_address_tokens"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# =========================================================
# SAVE
# =========================================================

missed_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\nV3 MISSED-MATCH ANALYSIS COMPLETE"
)