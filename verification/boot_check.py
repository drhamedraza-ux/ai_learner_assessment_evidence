import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT=Path("/home/claude/work/P02/PCDT_P2_runs/runs")
def paired_bootstrap(pa,pb,y,rows_idx,n_boot,seed):
    rg=np.random.default_rng(seed); learners=np.unique(rows_idx)
    by={l:np.where(rows_idx==l)[0] for l in learners}; d=[]
    for _ in range(n_boot):
        s=rg.choice(learners,len(learners),replace=True)
        ii=np.concatenate([by[l] for l in s])
        if len(np.unique(y[ii]))<2: continue
        d.append(roc_auc_score(y[ii],pa[ii])-roc_auc_score(y[ii],pb[ii]))
    d=np.asarray(d); lo,hi=np.percentile(d,[2.5,97.5])
    pv=2*min((d<=0).mean(),(d>=0).mean())
    return float(roc_auc_score(y,pa)-roc_auc_score(y,pb)),float(lo),float(hi),float(min(pv,1))
rd=ROOT/"RQ2RQ3__assistments_2021__seed0"
m=json.load(open(rd/"manifest.json")); z=np.load(rd/"artifacts/predictions_test.npz")
Y,M=z["targets"],z["mask"]>0
rows_idx=np.repeat(np.arange(z["PCDT"].shape[0])[:,None],z["PCDT"].shape[1],1)[M]
filed=pd.read_csv(rd/"tables/table3_statistics.csv").set_index("comparison")
out=[]
for k in ["DKVMN","Control"]:
    d,lo,hi,pv=paired_bootstrap(z["PCDT"][M],z[k][M],Y[M],rows_idx,m["config"]["n_bootstrap"],m["seed"])
    f=filed.loc[f"PCDT - {k}"]
    out.append(f"PCDT - {k:8s} recomputed delta={d:+.8f} ci=[{lo:+.6f},{hi:+.6f}] p={pv:.6f}")
    out.append(f"                 on file   delta={f.delta_auc:+.8f} ci=[{f.ci_lo:+.6f},{f.ci_hi:+.6f}] p={f.p:.6f}")
    ex=all(abs(a-b)<1e-9 for a,b in [(d,f.delta_auc),(lo,f.ci_lo),(hi,f.ci_hi),(pv,f.p)])
    out.append(f"                 EXACT MATCH (numpy 1.26 here vs 2.3 there): {ex}")
open("boot_check.txt","w").write("\n".join(out))
