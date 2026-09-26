import pandas as pd

GT_PATH = r"C:\riya\ML_Challenge\dataset\train\train_ground_truth.tsv"

gt = pd.read_csv(GT_PATH, sep="\t")

# Empty ground truth = no matches
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

# Count matches
gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if x == "" else len(x.split(","))
)

print("=" * 70)
print("GROUND TRUTH MATCH DISTRIBUTION")
print("=" * 70)

print("\nTotal S1 entities:", len(gt))

print("\nMatch count statistics:")
print(gt["match_count"].describe())

print("\nDistribution of number of matches:")
print(
    gt["match_count"]
    .value_counts()
    .sort_index()
    .head(30)
)

print("\nS1 entities with ZERO matches:",
      (gt["match_count"] == 0).sum())

print("S1 entities with >=1 match:",
      (gt["match_count"] >= 1).sum())

print("S1 entities with >=2 matches:",
      (gt["match_count"] >= 2).sum())

print("S1 entities with >=5 matches:",
      (gt["match_count"] >= 5).sum())

print("S1 entities with >=10 matches:",
      (gt["match_count"] >= 10).sum())


# --------------------------------------------------
# S2 vs S3
# --------------------------------------------------

def count_sources(x):
    if not x:
        return 0, 0

    ids = x.split(",")

    s2 = sum(i.startswith("S2-") for i in ids)
    s3 = sum(i.startswith("S3-") for i in ids)

    return s2, s3


source_counts = gt["matched_entity_ids"].apply(count_sources)

gt["s2_count"] = source_counts.apply(lambda x: x[0])
gt["s3_count"] = source_counts.apply(lambda x: x[1])

print("\nTotal S2 matches:", gt["s2_count"].sum())
print("Total S3 matches:", gt["s3_count"].sum())

print("\nS1 with only S2 matches:",
      ((gt["s2_count"] > 0) & (gt["s3_count"] == 0)).sum())

print("S1 with only S3 matches:",
      ((gt["s2_count"] == 0) & (gt["s3_count"] > 0)).sum())

print("S1 with BOTH S2 and S3 matches:",
      ((gt["s2_count"] > 0) & (gt["s3_count"] > 0)).sum())