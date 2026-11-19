import pandas as pd, numpy as np, sys
d=pd.read_parquet(sys.argv[1])
t=lambda s: pd.to_timedelta(s.where(s.str.match(r'^\d\d:\d\d'),None),errors='coerce')
d['dur']=((t(d.time_to)-t(d.time_from)).dt.total_seconds()/3600)
d.loc[d.dur<0,'dur']+=24
d['yr']=d.date.dt.year
ok=(abs(d.hrs-d.dur)<0.02)|((d.dur.fillna(0)==0)&(d.hrs<12))
d['hrs_clean']=np.where(ok,d.hrs,np.nan)
d['flag_hrs_mismatch']=~ok & (d.hrs>0)
d.to_parquet(sys.argv[2],index=False)
