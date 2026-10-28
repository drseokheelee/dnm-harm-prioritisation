"""Descriptive harm profile (RQ1): overall distribution, product classes,
prices and markets.

Usage: python descriptive_stats.py DATA_JSON OUT_DIR
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame(json.load(open(sys.argv[1], encoding="utf-8")))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
h = df["harm_score"].astype(float)
df["harmful"] = h >= 0.50
df["severe"] = h >= 0.90
df["price"] = pd.to_numeric(df["price"], errors="coerce")

print(f"Listings: {len(df)}")
print(f"Harmful (>= 0.50): {df['harmful'].sum()} ({100 * df['harmful'].mean():.1f}%)")
print(f"Most severe (>= 0.90): {df['severe'].sum()} ({100 * df['severe'].mean():.1f}%)")
print(f"Mean {h.mean():.3f}, median {h.median():.3f}")

cls = df.groupby("class").agg(listings=("harm_score", "size"), mean_harm=("harm_score", "mean"),
                              composite_risk=("harm_score", "sum"), harmful=("harmful", "sum"))
cls["share_pct"] = 100 * cls["listings"] / len(df)
cls = cls.sort_values("composite_risk", ascending=False)
cls.round(3).to_csv(f"{out}/product_classes.csv")
print("\nProduct classes\n", cls.round(3))

p = df[df["price"] > 0]
rho = spearmanr(p["price"], p["harm_score"]).correlation
print(f"\nPrices: n = {len(p)}, median {p['price'].median():.0f}, range {p['price'].min():.2f}-{p['price'].max():.0f}, Spearman rho = {rho:.2f}")

m = df[df["market_name"].notna() & (df["market_name"] != "")]
def largest(g):
    c = g["class"].value_counts()
    return f"{c.index[0]} ({100 * c.iloc[0] / len(g):.1f})"
mk = m.groupby("market_name").agg(listings=("harm_score", "size"), mean_harm=("harm_score", "mean"),
                                  sd_harm=("harm_score", "std"), harmful_pct=("harmful", "mean"),
                                  severe=("severe", "sum"), harm_mass=("harm_score", "sum"))
mk["harmful_pct"] *= 100
mk["largest_class"] = m.groupby("market_name").apply(largest)
mk = mk.sort_values("mean_harm")
mk.round(3).to_csv(f"{out}/markets.csv")
print("\nMarkets\n", mk.round(3))
