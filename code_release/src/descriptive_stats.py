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

# Concentration across markets and product classes
n_all, sev_all, mass_all, harm_all = len(df), df["severe"].sum(), h.sum(), df["harmful"].sum()
two = ["Yoga", "Anonymus-gun-shop"]
t = mk.loc[two]
print(f"\nYoga + Anonymus-gun-shop: {100 * t['listings'].sum() / n_all:.1f}% of listings, "
      f"{100 * t['severe'].sum() / sev_all:.1f}% of most severe listings, "
      f"{100 * t['harm_mass'].sum() / mass_all:.1f}% of total harm score "
      f"(Yoga {100 * mk.loc['Yoga', 'harm_mass'] / mass_all:.1f}%, "
      f"Anonymus-gun-shop {100 * mk.loc['Anonymus-gun-shop', 'harm_mass'] / mass_all:.1f}%)")
harmful_by_market = m.groupby("market_name")["harmful"].sum().sort_values(ascending=False)
top3 = harmful_by_market.head(3)
print(f"Top 3 markets by harmful listings ({', '.join(top3.index)}): "
      f"{100 * top3.sum() / harm_all:.1f}% of harmful listings")
dw = cls.loc[["drugs", "weapons"]]
print(f"Drugs + weapons: {100 * dw['composite_risk'].sum() / mass_all:.1f}% of total harm score, "
      f"{100 * dw['harmful'].sum() / harm_all:.1f}% of harmful listings")
print("Median price by class:", p.groupby("class")["price"].median().loc[["drugs", "weapons"]].to_dict())
