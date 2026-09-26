import pandas as pd
import sqlite3
import os
import re

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\riya\ML_Challenge"

NORMALIZED = os.path.join(BASE, "dataset", "normalized")
TRAIN = os.path.join(BASE, "dataset", "train")
VALIDATION = os.path.join(BASE, "dataset", "validation")

VALIDATION_IDS = os.path.join(
    VALIDATION, "validation_s1_ids.txt"
)

DB_FILE = os.path.join(
    VALIDATION, "v3_candidates.db"
)

OUTPUT_FILE = os.path.join(
    VALIDATION, "candidate_pairs_v3.tsv"
)

CHUNK_SIZE = 100000
SAMPLE_SIZE = 5000


# ============================================================
# BLOCKING FUNCTIONS
# ============================================================

def nums(x):
    if not isinstance(x, str):
        return []
    return re.findall(r"\b\d+[a-z]?\b", x.lower())


def toks(x):
    if not isinstance(x, str):
        return []
    return re.findall(r"[a-z]+", x.lower())


def name_prefix6(x):
    return x.replace(" ", "")[:6] if x else ""


def name_first_last(x):
    p = x.split() if x else []
    if not p:
        return ""
    if len(p) == 1:
        return p[0]
    return p[0] + "|" + p[-1]


def name_first(x):
    p = x.split() if x else []
    return p[0] if p else ""


def address_number(x):
    n = nums(x)
    return n[0] if n else ""


def address_number_token(x):
    n = nums(x)
    t = toks(x)

    if not n or not t:
        return ""

    return n[0] + "|" + t[0]


def address_prefix6(x):
    return x.replace(" ", "")[:6] if x else ""


def addr_number_two_tokens(x):
    n = nums(x)
    t = toks(x)

    if not n or len(t) < 2:
        return ""

    return n[0] + "|" + t[0] + "|" + t[1]


def addr_two_numbers(x):
    n = nums(x)

    if len(n) < 2:
        return ""

    return n[0] + "|" + n[1]


def addr_number_last_token(x):
    n = nums(x)
    t = toks(x)

    if not n or not t:
        return ""

    return n[0] + "|" + t[-1]


def addr_first_two_tokens(x):
    return "|".join(toks(x)[:2])


def addr_first_three_tokens(x):
    return "|".join(toks(x)[:3])


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
# LOAD VALIDATION IDS
# ============================================================

print("=" * 80)
print("BLOCKING V3 - SAVE CANDIDATES")
print("=" * 80)

print("\nLoading validation IDs...")

val_ids = pd.read_csv(
    VALIDATION_IDS,
    dtype=str
)

# Handle either possible column name
if "source1_entity_id" in val_ids.columns:
    val_ids = val_ids["source1_entity_id"]
else:
    val_ids = val_ids.iloc[:, 0]

val_ids = set(
    val_ids.head(SAMPLE_SIZE)
)

print("Validation S1:", len(val_ids))


# ============================================================
# LOAD ONLY VALIDATION S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    os.path.join(NORMALIZED, "train_source1.tsv"),
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

print("S1 records:", len(s1))


# ============================================================
# BUILD S1 LOOKUPS
# ============================================================

print("\nCreating S1 blocking keys...")

lookups = {
    rule: {}
    for rule in RULES
}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]

    for rule, function in RULES.items():

        if rule.startswith("name"):
            value = function(
                row["name_norm"]
            )
        else:
            value = function(
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
        f"{rule}: {len(lookups[rule]):,} keys"
    )


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

print("\nCreating disk-backed candidate database...")

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("PRAGMA temp_store=FILE")

conn.execute("""
CREATE TABLE candidates (
    source1_entity_id TEXT,
    candidate_entity_id TEXT,
    PRIMARY KEY (
        source1_entity_id,
        candidate_entity_id
    )
)
""")

conn.commit()


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(filename):

    print("\n" + "=" * 80)
    print("Processing:", filename)
    print("=" * 80)

    path = os.path.join(
        NORMALIZED,
        filename
    )

    total_rows = 0

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
        ),
        start=1
    ):

        total_rows += len(chunk)

        # IMPORTANT:
        # Only candidates generated from THIS chunk
        # are kept in RAM.
        pairs = set()

        for rule, function in RULES.items():

            if rule.startswith("name"):
                values = (
                    chunk["name_norm"]
                    .fillna("")
                    .map(function)
                )
            else:
                values = (
                    chunk["address_norm"]
                    .fillna("")
                    .map(function)
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

                s1_matches = lookups[
                    rule
                ].get(key)

                if not s1_matches:
                    continue

                other_id = chunk.at[
                    idx,
                    "entity_id"
                ]

                for s1_id in s1_matches:

                    pairs.add(
                        (
                            s1_id,
                            other_id
                        )
                    )

        if pairs:

            conn.executemany(
                """
                INSERT OR IGNORE INTO candidates
                VALUES (?, ?)
                """,
                pairs
            )

            conn.commit()

        if chunk_no % 5 == 0:
            print(
                f"  processed "
                f"{total_rows:,} rows",
                end="\r"
            )

        del pairs
        del chunk

    print(
        f"\nFinished {filename}: "
        f"{total_rows:,} rows scanned"
    )


# ============================================================
# SCAN S2 + S3
# ============================================================

process_source(
    "train_source2.tsv"
)

process_source(
    "train_source3.tsv"
)


# ============================================================
# COUNT CANDIDATES
# ============================================================

candidate_count = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print("\n" + "=" * 80)
print(
    f"UNIQUE CANDIDATE PAIRS: {candidate_count:,}"
)
print("=" * 80)


# ============================================================
# EXPORT TSV
# ============================================================

print("\nExporting candidate_pairs_v3.tsv...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

conn.execute("""
CREATE INDEX IF NOT EXISTS idx_s1
ON candidates(source1_entity_id)
""")

conn.commit()

first = True

for chunk in pd.read_sql_query(
    """
    SELECT
        source1_entity_id,
        candidate_entity_id
    FROM candidates
    ORDER BY source1_entity_id
    """,
    conn,
    chunksize=100000
):

    grouped = (
        chunk
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(
            lambda x: ",".join(
                sorted(set(x))
            )
        )
        .reset_index()
    )

    grouped.columns = [
        "source1_entity_id",
        "candidate_entity_ids"
    ]

    grouped.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first else "a",
        header=first
    )

    first = False

conn.close()

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDatabase:")
print(DB_FILE)

print("\nBLOCKING V3 COMPLETE")