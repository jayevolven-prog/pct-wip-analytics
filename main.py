"""Run the WIP pipeline on Railway: clean -> prep -> models -> load Postgres."""
import os, sys, glob, json, subprocess, pandas as pd
from sqlalchemy import create_engine, text

PY = sys.executable
os.makedirs("work/dash", exist_ok=True)
src = sorted(glob.glob("data/*.xlsx"))[-1]
print("source:", src, flush=True)
for args in (["pipeline/clean.py", src, "work/wip.parquet"],
             ["pipeline/prep.py", "work/wip.parquet", "work/wip2.parquet"],
             ["pipeline/models.py", "work/wip2.parquet", "work/results.json"],
             ["pipeline/build_dash.py", "work/wip2.parquet", "work/results.json", "work/dash"],
             ["pipeline/report.py", "work/wip2.parquet", "pipeline/stages.csv", "work/report"],
             ["pipeline/report2.py", "work/wip2.parquet", "pipeline/stages.csv", "work/report", "pipeline"],
             ["pipeline/report3.py", "work/wip2.parquet", "pipeline/stages.csv", "work/report", "pipeline"],
             ["pipeline/report4.py", "work/wip2.parquet", "work/report", "data/ext"],
             ["pipeline/report5.py", "work/wip2.parquet", "work/report", "data/ext"],
             ["pipeline/report6.py", "work/wip2.parquet", "work/report", "data/ext"],
             ["pipeline/report7.py", "work/wip2.parquet", "work/report", "data/ext"],
             ["pipeline/report8.py", "work/wip2.parquet", "work/report", "data/ext"],
             ["pipeline/report9.py", "work/wip2.parquet", "work/report", "data/ext"]):
    print("running", args[0], flush=True)
    subprocess.run([PY, "-I", *args], check=True)

eng = create_engine(os.environ["DATABASE_URL"].replace("postgres://", "postgresql://", 1).replace("postgresql://", "postgresql+psycopg2://", 1))
with eng.begin() as c:
    c.execute(text("DROP VIEW IF EXISTS machining_jobs")); c.execute(text("DROP VIEW IF EXISTS lost_time_by_month"))
df = pd.read_parquet("work/wip2.parquet")
df["flag_hrs_mismatch"] = df["flag_hrs_mismatch"].astype(bool)
df.to_sql("time_entries", eng, if_exists="replace", index=False, chunksize=1000, method="multi")
print("time_entries:", len(df), flush=True)
if os.path.exists("work/sales_orders.csv"):
    so = pd.read_csv("work/sales_orders.csv", parse_dates=["entered", "required", "promised"])
    so.to_sql("sales_orders", eng, if_exists="replace", index=False, chunksize=1000, method="multi"); print("sales_orders:", len(so), flush=True)
for pat, tab in (("data/ext/PCT_Core_Works_Orders_Continuum*.xlsx", "core_tasks"), ("data/ext/PCT_Order_Book_*.xlsx", "order_book"), ("data/ext/PCT_Pattern_Weights*.xlsx", "pattern_weights"), ("data/ext/PCT_Core_Schedule*.xlsx", "core_schedule"), ("data/ext/PCT_Production_Plan*.xlsx", "pct_plan")):
    for x in sorted(glob.glob(pat))[-1:]:
        t = pd.read_excel(x, sheet_name=0); t.to_sql(tab, eng, if_exists="replace", index=False); print(tab, len(t), flush=True)
for x in sorted(glob.glob("data/ext/PCT_Invoice_Analysis_*.xlsx"))[-1:]:
    for sh, t in pd.read_excel(x, sheet_name=None).items():
        if sh in ("Notes", "Check against PDF totals"): continue
        t.to_sql("invoice_" + sh.lower().replace(" ", "_"), eng, if_exists="replace", index=False); print("invoice_" + sh.lower().replace(" ", "_"), len(t), flush=True)
for x in sorted(glob.glob("data/ext/PCT_Scrap_Analysis_*.xlsx"))[-1:]:
    for sh, t in pd.read_excel(x, sheet_name=None).items():
        if sh in ("Notes", "Check against PDF totals"): continue
        t.to_sql("scrap_" + sh.lower().replace(" ", "_"), eng, if_exists="replace", index=False); print("scrap_" + sh.lower().replace(" ", "_"), len(t), flush=True)
for f in glob.glob("work/dash/*.json"):
    name = "agg_" + os.path.basename(f)[:-5]
    pd.DataFrame(json.load(open(f))["rows"]).to_sql(name, eng, if_exists="replace", index=False)
    print(name, flush=True)
for f in sorted(glob.glob("work/report/*.json")):
    name = "rpt_" + os.path.basename(f)[:-5]
    rows = json.load(open(f))["rows"]
    pd.DataFrame(rows).astype({"value": str} if name == "rpt_kpis" else {}).to_sql(name, eng, if_exists="replace", index=False)
    print(name, len(rows), flush=True)
with eng.begin() as c:
    c.execute(text("CREATE TABLE IF NOT EXISTS model_runs (run_at timestamptz DEFAULT now(), source text, results jsonb)"))
    c.execute(text("INSERT INTO model_runs (source, results) VALUES (:s, CAST(:r AS jsonb))"),
              {"s": os.path.basename(src), "r": open("work/results.json").read().replace("NaN", "null")})
    for sql in [
        "CREATE INDEX IF NOT EXISTS ix_te_date ON time_entries(date)",
        "CREATE INDEX IF NOT EXISTS ix_te_part ON time_entries(part_no)",
        "CREATE INDEX IF NOT EXISTS ix_te_wc ON time_entries(work_centre)",
        """CREATE OR REPLACE VIEW machining_jobs AS
           SELECT wo_no, part_no, work_centre, MIN(date) AS start_date, MAX(qty) AS qty,
                  SUM(hrs_clean) AS hrs, SUM(hrs_clean)/NULLIF(MAX(qty),0) AS hrs_per_unit
           FROM time_entries WHERE type='Booked' AND hrs_clean > 0
           GROUP BY wo_no, part_no, work_centre""",
        """CREATE OR REPLACE VIEW lost_time_by_month AS
           SELECT date_trunc('month', date) AS month, work_centre, lost_time, SUM(hrs_clean) AS hrs
           FROM time_entries WHERE type='Lost' GROUP BY 1,2,3""",
    ]:
        c.execute(text(sql))
print("done", flush=True)
