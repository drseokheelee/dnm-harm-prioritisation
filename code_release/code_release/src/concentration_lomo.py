"""Concentration of listings and harm across markets and product classes
(Gini coefficients), and leave-one-market-out evaluation of the ridge model
using text, category, keyword and price features (no image features).

Usage: python concentration_lomo.py DATA_JSON
Writes concentration_lomo_results.json and market_tab.csv to the working
directory.
"""
import json, sys, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.DataFrame(json.load(open(sys.argv[1], encoding="utf-8")))
df["title"]=df["title"].fillna(""); df["category"]=df["category"].fillna("unknown"); df["keyword"]=df["keyword"].fillna("unknown")
df["price"]=pd.to_numeric(df["price"],errors="coerce").fillna(0.0); y=df["harm_score"].astype(float).values
def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); c=np.cumsum(x)
    return (n+1-2*np.sum(c)/c[-1])/n
out={}
m=df[df["market_name"].notna() & (df["market_name"]!="")]
g=m.groupby("market_name")
tab=pd.DataFrame({"listings":g.size(),"harmful":g["harm_score"].apply(lambda s:(s>=0.5).sum()),
 "severe":g["harm_score"].apply(lambda s:(s>=0.9).sum()),"mass":g["harm_score"].sum()})
out["gini_market"]={c:float(gini(tab[c])) for c in tab}
gc=df.groupby("class")
tc=pd.DataFrame({"listings":gc.size(),"mass":gc["harm_score"].sum(),"severe":gc["harm_score"].apply(lambda s:(s>=0.9).sum())})
out["gini_class"]={c:float(gini(tc[c])) for c in tc}; out["n_class"]=len(tc)
# Lorenz-style: share of severe listings covered by top-k markets ranked by severe count
s=tab.sort_values("severe",ascending=False); out["cum_severe_topk"]=(s["severe"].cumsum()/s["severe"].sum()).round(3).tolist(); out["cum_list_topk"]=(s["listings"].cumsum()/tab["listings"].sum()).round(3).tolist(); out["order"]=s.index.tolist()
tab.to_csv("market_tab.csv")
# LOMO
def feats(fit,apply,bl):
    tf=TfidfVectorizer(max_features=500,ngram_range=(1,2)).fit(df["title"].iloc[fit])
    oc=OneHotEncoder(handle_unknown="ignore").fit(df[["category"]].iloc[fit]); ok=OneHotEncoder(handle_unknown="ignore").fit(df[["keyword"]].iloc[fit])
    ps=MinMaxScaler().fit(df[["price"]].iloc[fit]); r=[]
    for i in apply:
        d={"tfidf":tf.transform(df["title"].iloc[i]),"category":oc.transform(df[["category"]].iloc[i]),"keyword":ok.transform(df[["keyword"]].iloc[i]),"price":csr_matrix(np.clip(ps.transform(df[["price"]].iloc[i]),0,1))}
        r.append(hstack([d[b] for b in bl]).tocsr())
    return r
CF={"text_meta":["tfidf","category","keyword","price"],"cat_kw":["category","keyword"],"title":["tfidf"]}
lomo={k:{} for k in CF}; pooled={k:([],[]) for k in CF}
idx_all=np.arange(len(df)); mk=df["market_name"].fillna("")
for mkt in tab.index:
    te=idx_all[mk==mkt]; tr=idx_all[mk!=mkt]
    unseen_cat=float((~df["category"].iloc[te].isin(set(df["category"].iloc[tr]))).mean())
    for k,bl in CF.items():
        Xtr,Xte=feats(tr,[tr,te],bl); p=np.clip(Ridge(alpha=0.01).fit(Xtr,y[tr]).predict(Xte),0,1)
        lomo[k][mkt]={"n":int(len(te)),"MAE":float(mean_absolute_error(y[te],p)),"unseen_cat":unseen_cat,
          "sev_recall":float(((p>=0.9)&(y[te]>=0.9)).sum()/max(1,(y[te]>=0.9).sum()))}
        pooled[k][0].extend(y[te]); pooled[k][1].extend(p)
out["lomo"]=lomo
out["lomo_pooled"]={k:{"MAE":float(mean_absolute_error(a,b)),"R2":float(r2_score(a,b))} for k,(a,b) in pooled.items()}
json.dump(out,open("concentration_lomo_results.json","w"),indent=1)
print(json.dumps({k:out[k] for k in ["gini_market","gini_class","n_class","order","cum_severe_topk","cum_list_topk","lomo_pooled"]},indent=1))
for k in CF:
    print(k); [print(f"  {m:18s} n={v['n']:4d} MAE={v['MAE']:.3f} unseenCat={v['unseen_cat']:.2f} sevRec={v['sev_recall']:.2f}") for m,v in lomo[k].items()]
