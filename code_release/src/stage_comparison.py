"""Compare the Stage 1 rule-based scores with the final harm scores.

Applies harm_scorer.py (Stage 1) to the records of the final dataset and
reports how many final scores differ from the Stage 1 value, and how far the
final scores were assigned at the level of product keywords.

Note: the Stage 1 routing counts reported in the paper (harmful / non-harmful /
review) were produced by running harm_scorer.py on the raw parsed listings,
before market and category names were harmonised. Running it on the final
dataset gives slightly different routing counts.

Usage: python stage_comparison.py DATA_JSON
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harm_scorer import score_product

data = json.load(open(sys.argv[1], encoding="utf-8"))
df = pd.DataFrame({"keyword": [r.get("keyword") for r in data],
                   "cls": [r.get("class") for r in data],
                   "stage1": [score_product(dict(r))["harm_score"] for r in data],
                   "final": [float(r["harm_score"]) for r in data]})
df["changed"] = (df["stage1"] - df["final"]).abs() > 1e-3
n = len(df)
print(f"Listings: {n}")
print(f"Final score differs from Stage 1: {df['changed'].sum()} ({100 * df['changed'].mean():.1f}%)")
print("  by product class:", df[df["changed"]].groupby("cls").size().sort_values(ascending=False).to_dict())
print(f"Distinct final score values: {df['final'].nunique()}")

k = df.groupby("keyword")["final"].agg(["size", "nunique"])
single = k[k["nunique"] == 1]
print(f"Product keywords: {len(k)}; with a single final score: {len(single)}, "
      f"covering {single['size'].sum()} listings ({100 * single['size'].sum() / n:.1f}%)")
