import pandas as pd, requests, io, yfinance as yf, warnings
warnings.filterwarnings("ignore")
out={}
for sid in ["DGS10","DTB3","DGS20","DGS5"]:
    r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",timeout=60); r.raise_for_status()
    d=pd.read_csv(io.StringIO(r.text)); d.columns=["date",sid]; d["date"]=pd.to_datetime(d["date"]); d[sid]=pd.to_numeric(d[sid],errors="coerce")
    out[sid]=d.set_index("date")[sid]; print(sid,out[sid].first_valid_index().date(),out[sid].last_valid_index().date(),out[sid].notna().sum())
F=pd.DataFrame(out)
g=yf.download("^GSPC",start="1950-01-01",auto_adjust=False,progress=False)
g=g["Close"]["^GSPC"] if isinstance(g.columns,pd.MultiIndex) else g["Close"]
g.index=pd.to_datetime(g.index).tz_localize(None); print("GSPC",g.index[0].date(),g.index[-1].date(),len(g))
F["GSPC"]=g; F.to_parquet("long_data.parquet")
