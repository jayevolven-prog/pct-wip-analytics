import pandas as pd, numpy as np, sys
src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src, header=2)
df = df[pd.to_datetime(df['Date'], errors='coerce').notna()].copy()
df.columns = ['date','payroll','name','type','wo_no','part_no','qty','work_centre','rate_gbp_hr','lost_time','time_from','time_to','hrs','notes']
df['date'] = pd.to_datetime(df['date'])
df['wo_base'] = df['wo_no'].astype(str).str.split('/').str[0]
df['wo_op'] = df['wo_no'].astype(str).str.split('/').str[1]
df['part_no'] = df['part_no'].astype(str).str.strip()
df['labour_gbp'] = df['hrs'] * df['rate_gbp_hr']
df['lost_code'] = df['lost_time'].str.extract(r'^(\d+)').astype(float)
df['time_from'] = df['time_from'].astype(str); df['time_to'] = df['time_to'].astype(str)
df.to_parquet(out, index=False)
print(df.shape); print(df[['wo_op']].value_counts().head(10))
print(df.groupby('work_centre')['wo_op'].agg(lambda s: s.value_counts().index[0]).head(40).to_string())
