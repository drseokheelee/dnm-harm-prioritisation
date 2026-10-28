"""Create the three 70/15/15 train/validation/test splits used in the paper.

Usage: python make_splits.py DATA_JSON OUT_JSON
"""
import json, sys
import numpy as np
from sklearn.model_selection import train_test_split

data = json.load(open(sys.argv[1], encoding="utf-8"))
idx = np.arange(len(data))
splits = {}
for seed in (0, 1, 2):
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    splits[str(seed)] = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
json.dump(splits, open(sys.argv[2], "w"))
print({s: {k: len(v) for k, v in d.items()} for s, d in splits.items()})
