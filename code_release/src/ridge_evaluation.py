"""Ridge regression on text, category, keyword, price and image features:
training/validation/test performance, the feature-group ablation and an
alpha robustness check, for each of the three splits.

Run after cnn_embeddings.py.
Usage: python ridge_evaluation.py DATA_JSON SPLITS_JSON EMB_DIR OUT_JSON
"""
import json, sys
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

DATA, SPLITS, EMB, OUTJ = sys.argv[1:5]
df = pd.DataFrame(json.load(open(DATA, encoding="utf-8")))
df["title"] = df["title"].fillna(""); df["category"] = df["category"].fillna("unknown")
df["keyword"] = df["keyword"].fillna("unknown")
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
y = df["harm_score"].astype(float).values; N = len(df)
splits = json.load(open(SPLITS))
ALPHA, K = 0.01, 32
BLOCKS = ["tfidf", "category", "keyword", "price"]
ABL = [("full", BLOCKS, True), ("no_image", BLOCKS, False), ("image_only", [], True),
       ("category+keyword", ["category", "keyword"], False), ("category_only", ["category"], False),
       ("title_tfidf_only", ["tfidf"], False), ("price_only", ["price"], False)]

def tab(fit, apply):
    tf = TfidfVectorizer(max_features=500, ngram_range=(1, 2)).fit(df["title"].iloc[fit])
    oc = OneHotEncoder(handle_unknown="ignore").fit(df[["category"]].iloc[fit])
    ok = OneHotEncoder(handle_unknown="ignore").fit(df[["keyword"]].iloc[fit])
    ps = MinMaxScaler().fit(df[["price"]].iloc[fit])
    return [{"tfidf": tf.transform(df["title"].iloc[i]), "category": oc.transform(df[["category"]].iloc[i]),
             "keyword": ok.transform(df[["keyword"]].iloc[i]),
             "price": csr_matrix(np.clip(ps.transform(df[["price"]].iloc[i]), 0, 1))} for i in apply]

def img(E, fit, apply):
    sc = StandardScaler().fit(E[fit]); p = PCA(n_components=K, random_state=0).fit(sc.transform(E[fit]))
    return [p.transform(sc.transform(E[i])) for i in apply], float(p.explained_variance_ratio_.sum())

def X(T, Z, bl, im): return hstack([T[b] for b in bl] + ([csr_matrix(Z)] if im else [])).tocsr()
def met(a, b): return {"MAE": float(mean_absolute_error(a, b)), "R2": float(r2_score(a, b))}

res, preds = {}, {}
for s in ["0", "1", "2"]:
    tr, va, te = (np.array(splits[s][k]) for k in ("train", "val", "test"))
    trva = np.concatenate([tr, va]); E = np.load(f"{EMB}/embeddings_seed{s}.npy")
    R = {}
    T = tab(tr, [tr, va, te]); Z, var = img(E, tr, [tr, va, te])
    m = Ridge(alpha=ALPHA).fit(X(T[0], Z[0], BLOCKS, True), y[tr])
    for n, k in [("train", 0), ("val", 1), ("test_initial", 2)]:
        R[n] = met(y[[tr, va, te][k]], m.predict(X(T[k], Z[k], BLOCKS, True)))
    R["pca_var"] = var
    T2 = tab(trva, [trva, te]); Z2, var2 = img(E, trva, [trva, te]); R["pca_var_final"] = var2
    R["ablation"] = {}
    for n, bl, im in ABL:
        mm = Ridge(alpha=ALPHA).fit(X(T2[0], Z2[0], bl, im), y[trva])
        p = mm.predict(X(T2[1], Z2[1], bl, im)); R["ablation"][n] = met(y[te], p)
        if n == "full": preds[s] = {"idx": te.tolist(), "y": y[te].tolist(), "pred": p.tolist()}
    cv = RidgeCV(alphas=np.logspace(-4, 3, 15), cv=5).fit(X(T2[0], Z2[0], BLOCKS, True), y[trva])
    R["ridgecv"] = {"alpha": float(cv.alpha_), **met(y[te], cv.predict(X(T2[1], Z2[1], BLOCKS, True)))}
    res[s] = R
summ = {}
for n, _, _ in ABL:
    a = np.array([[res[s]["ablation"][n]["MAE"], res[s]["ablation"][n]["R2"]] for s in res])
    summ[n] = {"MAE_mean": a[:, 0].mean(), "MAE_sd": a[:, 0].std(ddof=1), "R2_mean": a[:, 1].mean(), "R2_sd": a[:, 1].std(ddof=1)}
res["summary"] = summ; res["preds"] = preds
json.dump(res, open(OUTJ, "w"), indent=1)
for s in ["0", "1", "2"]:
    r = res[s]; print(f"seed {s}: train {r['train']} | val {r['val']} | test_init {r['test_initial']} | final {r['ablation']['full']} | pca {r['pca_var']:.3f} | cv {r['ridgecv']}")
for n, v in summ.items(): print(f"{n:18s} MAE {v['MAE_mean']:.4f}±{v['MAE_sd']:.4f}  R2 {v['R2_mean']:.4f}±{v['R2_sd']:.4f}")
