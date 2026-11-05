import pandas as pd
from pathlib import Path

folder = Path("data/processed/demo_road/splits")

files = [
    "labeled_10pct.csv",
    "unlabeled_10pct.csv",
    "val.csv",
    "test.csv",
]

for name in files:
    df = pd.read_csv(folder / name)
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Unique source-destination pairs:", len(df[["src", "dst"]].drop_duplicates()))

def pair_set(filename):
    df = pd.read_csv(folder / filename)
    return set(zip(df["src"], df["dst"]))

labeled = pair_set("labeled_10pct.csv")
unlabeled = pair_set("unlabeled_10pct.csv")
validation = pair_set("val.csv")
test = pair_set("test.csv")

print("\n--- OVERLAP CHECK ---")
print("Labeled vs unlabeled:", len(labeled & unlabeled))
print("Unlabeled vs validation:", len(unlabeled & validation))
print("Unlabeled vs test:", len(unlabeled & test))
print("Validation vs test:", len(validation & test))
