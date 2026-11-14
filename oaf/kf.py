"""Ken French library loaders (survivorship-free portfolio data). Returns in decimal."""
import io, numpy as np, pandas as pd
D = "data/kf/"

def _blocks(path):
    txt = open(path, encoding="latin-1").read().splitlines(); blocks = []; cur = None
    for ln in txt:
        p = ln.split(",")
        if len(p) >= 2 and p[0].strip() == "" and not p[1].strip().replace(".", "").replace("-", "").isdigit():
            cur = dict(cols=[c.strip() for c in p[1:]], rows=[]); blocks.append(cur); continue
        if cur is not None and p[0].strip().isdigit() and len(p[0].strip()) >= 6:
            cur["rows"].append(p)
        elif cur is not None and cur["rows"] and p[0].strip() == "": cur = None
    out = []
    for b in blocks:
        if not b["rows"]: continue
        idx = pd.to_datetime([r[0].strip() for r in b["rows"]], format="%Y%m%d")
        a = np.array([[float(x) for x in r[1:1+len(b["cols"])]] for r in b["rows"]])
        a[a <= -99] = np.nan; out.append(pd.DataFrame(a / 100.0, idx, b["cols"]))
    return out

def industries(weighting="vw"):
    bl = _blocks(D + "49_Industry_Portfolios_Daily.csv"); return bl[0] if weighting == "vw" else bl[1]

def factors():
    f = _blocks(D + "F-F_Research_Data_Factors_daily.csv")[0]
    mom = _blocks(D + "F-F_Momentum_Factor_daily.csv")[0]; mom.columns = ["UMD"]
    return f.join(mom, how="left")

def factors8():
    """MKT-RF,SMB,HML,RMW,CMA,RF (2x3 5-factor, from 1963) + UMD + ST_Rev + LT_Rev; plus 3-factor (from 1926) for MKT/SMB/HML."""
    f3 = factors()
    f5 = _blocks(D + "F-F_Research_Data_5_Factors_2x3_daily.csv")[0]
    st = _blocks(D + "F-F_ST_Reversal_Factor_daily.csv")[0]; st.columns = ["STR"]
    lt = _blocks(D + "F-F_LT_Reversal_Factor_Daily.csv")[0]; lt.columns = ["LTR"]
    out = f3.copy(); out["RMW"] = f5["RMW"]; out["CMA"] = f5["CMA"]
    return out.join(st, how="left").join(lt, how="left")
