"""PCT manufacturing performance analytics.

Usage: python report.py <wip2.parquet> <stages.csv> <out_dir>
Writes one JSON file per report section to out_dir ({"rows": [...], "meta": [...]}).
Every section is stage-agnostic: stages come from stages.csv, so new BOM scan
points only need a new row there.
"""
import sys, os, json, math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SRC, STAGES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = open(os.path.abspath(__file__)).read()
SHIFT_HRS_WK = 39.0          # PCT week: 8.5 h Mon-Thu, 6 h Fri
DAY_HRS = {0: 8.5, 1: 8.5, 2: 8.5, 3: 8.5, 4: 6.0}
IDEAL_Q = 0.20               # best-demonstrated rate = 20th percentile hrs/unit
KPI = []

def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (pd.Timestamp,)): return v.strftime("%Y-%m-%d")
    return v

def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [{"label": "report.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": "WIP_REPORT_2021-2026.xlsx, Time Entry Details"}]
    if note: meta.append({"label": "Note", "value": note})
    json.dump({"rows": rows, "meta": meta}, open(f"{OUT}/{name}.json", "w"), allow_nan=False)
    print("wrote", name, len(rows), flush=True)

def kpi(key, value, label, unit=""):
    KPI.append({"key": key, "value": clean(value), "label": label, "unit": unit})

# ---------------------------------------------------------------- load
d = pd.read_parquet(SRC)
d["op"] = pd.to_numeric(d["wo_op"], errors="coerce")
st = pd.read_csv(STAGES)
def stage_of(op):
    if pd.isna(op): return (None, None, None)
    m = st[(st.op_from <= op) & (st.op_to >= op)]
    return (None, None, None) if m.empty else tuple(m.iloc[0][["stage", "stage_order", "flow"]])
smap = {op: stage_of(op) for op in d["op"].dropna().unique()}
d["stage"] = d["op"].map(lambda o: smap.get(o, (None,))[0])
d["stage_order"] = d["op"].map(lambda o: smap.get(o, (None, None))[1])
d["flow"] = d["op"].map(lambda o: smap.get(o, (None, None, None))[2])
END = d.date.max()
kpi("data_end", END.strftime("%d %b %Y"), "Data to")
kpi("entries", len(d), "Time entries analysed")
kpi("works_orders", d.wo_base.nunique(), "Works orders")
kpi("parts", d.part_no.nunique(), "Part numbers")

# machining centres = centres whose works-order bookings are mostly machining ops (100-269)
wcop = d[d.op.notna()].groupby("work_centre").op.agg(lambda s: s.between(100, 269).mean())
MSET = set(wcop[wcop > 0.5].index)
mc = d[(d.hrs_clean > 0) & d.work_centre.isin(MSET)].copy()   # machining with time, incl. lost time
MCS = mc.work_centre.value_counts()
MCS = list(MCS[MCS >= 50].index)

# ---------------------------------------------------------------- quality yield (qty first op -> despatch)
def yields(flow, first_ops, desp_ops):
    q = d[d.flow == flow].groupby(["wo_base", "op"]).qty.sum().unstack()
    f = q[[c for c in q.columns if c in first_ops]].bfill(axis=1).iloc[:, 0] if any(c in q.columns for c in first_ops) else None
    lst = q[[c for c in q.columns if c in desp_ops]].sum(axis=1, min_count=1)
    y = pd.DataFrame({"first": f, "desp": lst}).dropna()
    y = y[y["first"] > 0]
    start = d.groupby("wo_base").date.min()
    y["yr"] = start.reindex(y.index).dt.year
    y = y[start.reindex(y.index) < END - pd.Timedelta(days=45)]
    y["ratio"] = (y.desp / y["first"]).clip(upper=1)
    return y
ycast = yields("casting", [10], [80, 81, 82, 85, 90])
ymach = yields("machining", [110, 100, 105, 120], [280, 281, 285, 290])
qual = []
for yr in range(2021, 2027):
    for nm, y in (("Casting", ycast), ("Machining", ymach)):
        g = y[y.yr == yr]
        if len(g) > 20:
            qual.append({"year": str(yr), "flow": nm, "orders": len(g),
                         "qty_yield_pct": 100 * (g.desp.clip(upper=g["first"]).sum() / g["first"].sum()),
                         "orders_short_pct": 100 * (g.ratio < 1).mean()})
save("quality_yield", qual, "Quantity despatched / quantity started per works order, capped at 100%. Orders started in the last 45 days excluded.")
Q_MACH = ymach.desp.clip(upper=ymach["first"]).sum() / ymach["first"].sum()
Q_CAST = ycast.desp.clip(upper=ycast["first"]).sum() / ycast["first"].sum()
kpi("q_cast", 100 * Q_CAST, "Casting quantity yield", "%")
kpi("q_mach", 100 * Q_MACH, "Machining quantity yield", "%")

# ---------------------------------------------------------------- OEE (machining centres)
jobs = mc[mc.type == "Booked"].groupby(["wo_no", "part_no", "work_centre"]).agg(
    hrs=("hrs_clean", "sum"), qty=("qty", "max"), date=("date", "min")).reset_index()
jobs = jobs[jobs.qty > 0]
jobs["hpu"] = jobs.hrs / jobs.qty
cnt = jobs.groupby(["part_no", "work_centre"]).hpu.transform("size")
jobs["ideal_hpu"] = jobs.groupby(["part_no", "work_centre"]).hpu.transform(lambda s: s.quantile(IDEAL_Q))
jobs.loc[cnt < 4, "ideal_hpu"] = np.nan
jobs["ideal_hrs"] = jobs.ideal_hpu * jobs.qty
jobs["yr"] = jobs.date.dt.year
jobs["q"] = jobs.date.dt.to_period("Q").astype(str)
mc["q"] = mc.date.dt.to_period("Q").astype(str)

def oee_block(tm, jb, qrate):
    run = tm.loc[tm.type == "Booked", "hrs_clean"].sum()
    setup = tm.loc[tm.type == "Setup", "hrs_clean"].sum()
    lost = tm.loc[tm.type == "Lost", "hrs_clean"].sum()
    rew = tm.loc[tm.type == "Rework", "hrs_clean"].sum()
    planned = run + setup + lost + rew
    cov = jb[jb.ideal_hrs.notna()]
    perf = cov.ideal_hrs.sum() / cov.hrs.sum() if cov.hrs.sum() > 0 else np.nan
    avail = run / planned if planned > 0 else np.nan
    return dict(recorded_hrs=planned, run_hrs=run, setup_hrs=setup, lost_hrs=lost,
                availability=100 * avail, performance=100 * perf, quality=100 * qrate,
                oee=100 * avail * perf * qrate, perf_coverage_pct=100 * cov.hrs.sum() / max(jb.hrs.sum(), 1e-9))

o_year = []
for yr in range(2021, 2026):
    r = oee_block(mc[mc.yr == yr], jobs[jobs.yr == yr], ymach[ymach.yr == yr].pipe(lambda g: g.desp.clip(upper=g["first"]).sum() / g["first"].sum()))
    r["year"] = str(yr); o_year.append(r)
save("oee_year", o_year, "Machining centres, clean hours. Availability = run / (run + setup + lost + rework). Performance = ideal hrs / actual run hrs, ideal = 20th percentile hrs per unit for part and centre (parts with 4+ jobs). Quality = quantity yield to despatch. 2026 has no time data.")
o_all = oee_block(mc[mc.yr <= 2025], jobs[jobs.yr <= 2025], Q_MACH)
for k in ("availability", "performance", "quality", "oee"):
    kpi("oee_" + k, o_all[k], {"oee": "Machining OEE"}.get(k, k.title()), "%")
kpi("oee_cov", o_all["perf_coverage_pct"], "Run hours covered by a performance standard", "%")

o_wc = []
for wc in MCS:
    r = oee_block(mc[(mc.work_centre == wc) & (mc.yr <= 2025)], jobs[(jobs.work_centre == wc) & (jobs.yr <= 2025)], Q_MACH)
    if r["recorded_hrs"] >= 300:
        r["work_centre"] = wc; o_wc.append(r)
o_wc = sorted(o_wc, key=lambda r: -(r["oee"] if r["oee"] == r["oee"] else -1))
save("oee_centre", o_wc, "2021 to 2025, centres with 300+ recorded hours.")
best = max(o_wc, key=lambda r: r["oee"] or 0); worst = min(o_wc, key=lambda r: r["oee"] or 999)
kpi("oee_best_wc", best["work_centre"], "Highest OEE centre"); kpi("oee_best", best["oee"], "Highest centre OEE", "%")
kpi("oee_worst_wc", worst["work_centre"], "Lowest OEE centre"); kpi("oee_worst", worst["oee"], "Lowest centre OEE", "%")

o_q = []
for q in sorted(mc[mc.yr <= 2025].q.unique()):
    r = oee_block(mc[mc.q == q], jobs[jobs.q == q], Q_MACH)
    if r["recorded_hrs"] > 100: r["quarter"] = q; o_q.append(r)
save("oee_quarter", o_q, "Quarterly OEE, quality held at the 2021-2025 machining yield.")

# loss waterfall (hours, 2021-2025)
lt = mc[(mc.type == "Lost") & (mc.yr <= 2025)].groupby("lost_time").hrs_clean.sum().sort_values(ascending=False)
cov_all = jobs[(jobs.yr <= 2025) & jobs.ideal_hrs.notna()]
speed_loss = (cov_all.hrs.sum() - cov_all.ideal_hrs.sum()) * (o_all["run_hrs"] / cov_all.hrs.sum())
qual_loss = (o_all["run_hrs"] - speed_loss) * (1 - Q_MACH)
wf = [{"step": "Recorded machine time", "hrs": o_all["recorded_hrs"], "kind": "total"},
      {"step": "Setup and changeover", "hrs": -o_all["setup_hrs"], "kind": "loss"}]
for i, (k, v) in enumerate(lt.items()):
    if i < 3: wf.append({"step": "Lost: " + k.split(" - ", 1)[-1].replace("(A) ", "").replace("(N) ", "").replace("(K) ", ""), "hrs": -v, "kind": "loss"})
wf.append({"step": "Lost: other reasons", "hrs": -lt.iloc[3:].sum(), "kind": "loss"})
wf.append({"step": "Rework", "hrs": -(o_all["recorded_hrs"] - o_all["run_hrs"] - o_all["setup_hrs"] - o_all["lost_hrs"]), "kind": "loss"})
wf.append({"step": "Speed loss vs best rate", "hrs": -speed_loss, "kind": "loss"})
wf.append({"step": "Quantity not despatched", "hrs": -qual_loss, "kind": "loss"})
wf.append({"step": "Value-adding time (OEE)", "hrs": o_all["recorded_hrs"] - o_all["setup_hrs"] - lt.sum() - speed_loss - qual_loss - (o_all["recorded_hrs"] - o_all["run_hrs"] - o_all["setup_hrs"] - o_all["lost_hrs"]), "kind": "total"})
save("oee_waterfall", wf, "Hours 2021-2025. Speed loss scaled from covered jobs to all run hours.")
kpi("speed_loss_hrs", speed_loss, "Hours lost to running below best demonstrated rate", "hrs")
kpi("setup_hrs_total", o_all["setup_hrs"], "Setup hours 2021-2025", "hrs")

# capacity loading against one shift
lo = []
weeks = mc[mc.yr == 2024].date.dt.isocalendar().week.nunique()
for wc in MCS:
    h = mc[(mc.work_centre == wc) & (mc.yr == 2024)].hrs_clean.sum()
    if h > 0: lo.append({"work_centre": wc, "hrs_per_week": h / weeks, "loading_pct": 100 * h / weeks / SHIFT_HRS_WK})
save("loading", sorted(lo, key=lambda r: -r["loading_pct"]), f"2024 recorded hours per week vs {SHIFT_HRS_WK:.0f} h single-shift capacity (assumption). Under-recording means true load is higher.")

# ---------------------------------------------------------------- labour productivity
tt = mc[mc.time_from != "00:00:00"].copy()
tt["f"] = pd.to_timedelta(tt.time_from).dt.total_seconds() / 3600
tt["t"] = pd.to_timedelta(tt.time_to).dt.total_seconds() / 3600
tt.loc[tt.t < tt.f, "t"] += 24
def union(g):
    iv = sorted(zip(g.f, g.t)); tot = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + ce - cs
pdy = tt.groupby(["date", "name"]).apply(lambda g: pd.Series({"booked": g.hrs_clean.sum(), "clock": union(g)})).reset_index()
pdy["yr"] = pdy.date.dt.year
bc = pdy.groupby("yr")[["booked", "clock"]].sum()
# chained Laspeyres hrs-per-unit index on common parts
py = jobs[jobs.yr <= 2025].groupby(["yr", "part_no"]).agg(hrs=("hrs", "sum"), qty=("qty", "sum"))
py["hpu"] = py.hrs / py.qty
idx = {2021: 100.0}; prod = []
for yr in range(2022, 2026):
    a, b = py.loc[yr - 1], py.loc[yr]
    com = a.index.intersection(b.index)
    ratio = (b.loc[com, "hpu"] * a.loc[com, "qty"]).sum() / (a.loc[com, "hpu"] * a.loc[com, "qty"]).sum()
    idx[yr] = idx[yr - 1] * ratio
for yr in range(2021, 2026):
    prod.append({"year": str(yr), "hpu_index": idx[yr], "booked_per_clock": bc.loc[yr, "booked"] / bc.loc[yr, "clock"] if yr in bc.index else None})
save("productivity", prod, "Hours-per-unit index, chained Laspeyres on parts made in consecutive years, 2021 = 100. Lower is better.")
kpi("hpu_index_2025", idx[2025], "Mix-adjusted hours per unit index 2025 (2021 = 100)")
kpi("booked_clock_2025", bc.loc[2025, "booked"] / bc.loc[2025, "clock"], "Booked hours per clock hour 2025")

# mixed-effects learning / batch model
import statsmodels.formula.api as smf
mj = jobs[(jobs.yr <= 2025) & (jobs.hpu > 0)].copy()
mj = mj[mj.groupby("part_no").part_no.transform("size") >= 3]
mj["lhpu"] = np.log(mj.hpu); mj["lq"] = np.log(mj.qty); mj["year"] = mj.yr.astype(str)
mm = smf.mixedlm("lhpu ~ lq + C(year) + C(work_centre)", mj, groups=mj["part_no"]).fit(method="lbfgs")
ci = mm.conf_int()
me = [{"term": "Batch size elasticity b", "estimate": mm.params["lq"], "ci_low": ci.loc["lq", 0], "ci_high": ci.loc["lq", 1], "p": mm.pvalues["lq"],
       "meaning": f"Doubling batch size changes hours per unit by {100 * (2 ** mm.params['lq'] - 1):.0f}%"}]
for yr in range(2022, 2026):
    k = f"C(year)[T.{yr}]"
    me.append({"term": f"Year {yr} vs 2021", "estimate": mm.params[k], "ci_low": ci.loc[k, 0], "ci_high": ci.loc[k, 1], "p": mm.pvalues[k],
               "meaning": f"Hours per unit {100 * (math.exp(mm.params[k]) - 1):+.0f}% vs 2021, same part, centre and batch"})
save("mixed_model", me, f"Linear mixed model, log(hrs/unit) ~ log(qty) + year + work centre + random intercept per part. n = {len(mj)} jobs, {mj.part_no.nunique()} parts.")
kpi("batch_elasticity", mm.params["lq"], "Batch size elasticity")
kpi("batch_double_pct", 100 * (2 ** mm.params["lq"] - 1), "Change in hrs/unit when batch doubles", "%")
kpi("year2025_pct", 100 * (math.exp(mm.params["C(year)[T.2025]"]) - 1), "Like-for-like hrs/unit 2025 vs 2021", "%")
kpi("mm_jobs", len(mj), "Jobs in mixed model")

# setup amortisation: batch for setup <= 10% of job
su = mc[mc.type == "Setup"].groupby("wo_no").hrs_clean.sum()
jobs["setup"] = jobs.wo_no.map(su)
eb = []
for p, g in jobs[jobs.yr <= 2025].groupby("part_no"):
    g2 = g[g.setup > 0]
    if len(g2) >= 4:
        S, r = g2.setup.median(), g.hpu.median()
        if r > 0: eb.append({"part_no": p, "jobs": len(g), "setup_hrs": S, "run_hpu": r, "median_batch": g.qty.median(), "batch_for_10pct_setup": 9 * S / r})
eb = sorted(eb, key=lambda x: -x["jobs"])[:12]
save("batch_economics", eb, "Batch for setup <= 10% of job hours: Q = 9S / r, S = median setup hrs, r = median run hrs per unit.")

# ---------------------------------------------------------------- flow / process mining
ev = d[d.stage.notna()].groupby(["wo_base", "stage", "stage_order", "flow"]).agg(first=("date", "min"), last=("date", "max"), qty=("qty", "sum")).reset_index()
ev = ev.sort_values(["wo_base", "first", "stage_order"])
var = ev.groupby("wo_base").stage.apply(lambda s: " > ".join(s)).rename("route")
wo = ev.groupby("wo_base").agg(start=("first", "min"), end=("last", "max"), flow=("flow", "first"), n_stages=("stage", "nunique"))
wo = wo.join(var)
wo["lead_days"] = (wo.end - wo.start).dt.days
wo["despatched"] = wo.route.str.contains("despatch")
vv = wo.groupby("route").agg(orders=("lead_days", "size"), median_days=("lead_days", "median")).sort_values("orders", ascending=False)
vv["share_pct"] = 100 * vv.orders / vv.orders.sum()
save("variants", vv.head(10).reset_index(), "Process variants: sequence of stages per works order by first booking date (van der Aalst process mining).")
kpi("variants_n", len(vv), "Distinct routes")
kpi("top3_variant_share", vv.share_pct.head(3).sum(), "Share of orders on the top 3 routes", "%")
# directly-follows
dfg = []
for w, g in ev.groupby("wo_base"):
    s = g.stage.tolist(); f = g["first"].tolist()
    for i in range(len(s) - 1):
        dfg.append((s[i], s[i + 1], (f[i + 1] - f[i]).days))
dfg = pd.DataFrame(dfg, columns=["from", "to", "days"])
dg = dfg.groupby(["from", "to"]).days.agg(transitions="size", median_days="median", p90_days=lambda s: s.quantile(.9)).reset_index().sort_values("transitions", ascending=False)
save("transitions", dg.head(14), "Directly-follows relations between stages with waiting time in calendar days.")
# stage timeline for the two main routes
tl = []
for flow, route in (("casting", ["Mould and cast", "Fettle", "Casting despatch"]), ("machining", ["Machining op 1", "Machining op 2", "Machined despatch"])):
    cum = 0.0
    for a, b in zip(route[:-1], route[1:]):
        m = dg[(dg["from"] == a) & (dg["to"] == b)]
        if len(m):
            md = float(m.median_days.iloc[0]); tl.append({"flow": flow, "from": a, "to": b, "start_day": cum, "median_days": md, "p90_days": float(m.p90_days.iloc[0])}); cum += md
save("stage_timeline", tl)
# lead time by year and flow
lty = wo[wo.despatched & (wo.n_stages >= 2)].assign(yr=lambda x: x.start.dt.year).groupby(["yr", "flow"]).lead_days.agg(orders="size", median="median", p90=lambda s: s.quantile(.9), cv=lambda s: s.std() / s.mean()).reset_index()
lty["year"] = lty.yr.astype(str)
save("lead_time", lty.drop(columns="yr"), "Calendar days from first to last booking, despatched orders.")
for fl in ("casting", "machining"):
    g = lty[lty.flow == fl]
    kpi(f"lead_{fl}_2021", g[g.year == "2021"]["median"].iloc[0], f"Median {fl} lead time 2021", "days")
    kpi(f"lead_{fl}_2026", g[g.year == "2026"]["median"].iloc[0], f"Median {fl} lead time 2026", "days")
# WIP and Little's law (2025)
days = pd.date_range("2021-01-04", END, freq="D")
wip_rows = []
for fl in ("casting", "machining"):
    w = wo[wo.flow == fl]
    s = pd.Series(0, index=days)
    for a, b in zip(w.start, w.end): s[a:b] += 1
    wk = s.resample("W").mean()
    for t, v in wk.items(): wip_rows.append({"week": t.strftime("%Y-%m-%d"), "flow": fl, "open_orders": v})
    w25 = w[(w.end.dt.year == 2025) & w.despatched]
    L = s["2025"].mean(); lam = len(w25) / 365; W = w25.lead_days.mean()
    kpi(f"little_{fl}_L", L, f"Average open {fl} orders 2025"); kpi(f"little_{fl}_lw", lam * W, f"Little's law estimate {fl} 2025")
save("wip_weekly", wip_rows, "Average open works orders per week (first to last booking).")

# ---------------------------------------------------------------- survival (Kaplan-Meier + Cox)
from lifelines import KaplanMeierFitter, CoxPHFitter
sv = wo.copy()
sv["event"] = sv.despatched.astype(int)
sv["dur"] = np.where(sv.despatched, sv.lead_days, (END - sv.start).dt.days).clip(min=0) + 0.5
sv = sv[(sv.dur > 0) & ~((sv.event == 0) & ((END - sv.start).dt.days > 365))]   # drop stale never-despatched
km_rows = []
for fl in ("casting", "machining"):
    g = sv[sv.flow == fl]
    k = KaplanMeierFitter().fit(g.dur, g.event)
    for t in range(0, 121, 2):
        km_rows.append({"day": t, "flow": fl, "pct_open": 100 * float(k.predict(t))})
    kpi(f"km_{fl}_median", k.median_survival_time_, f"Kaplan-Meier median {fl} lead time", "days")
save("km_curves", km_rows, "Share of works orders still open after N days (Kaplan-Meier, open orders censored).")
qty_wo = d.groupby("wo_base").qty.max()
rep = d.groupby("part_no").wo_base.nunique()
part_wo = d.groupby("wo_base").part_no.first()
cx = sv.assign(log_qty=np.log1p(qty_wo.reindex(sv.index)), machining=(sv.flow == "machining").astype(int),
               extra_stages=sv.n_stages - 1, log_repeats=np.log1p(part_wo.reindex(sv.index).map(rep)),
               start_year=sv.start.dt.year - 2021, month_end_start=(sv.start.dt.day > 21).astype(int))
cols = ["log_qty", "machining", "extra_stages", "log_repeats", "start_year", "month_end_start"]
cph = CoxPHFitter(penalizer=0.01).fit(cx[cols + ["dur", "event"]].dropna(), "dur", "event")
s = cph.summary
lab = {"log_qty": "Batch size (x2.7)", "machining": "Machining order (vs casting)", "extra_stages": "Each extra stage",
       "log_repeats": "Repeat part (x2.7 orders)", "start_year": "Each later start year", "month_end_start": "Started in last 10 days of month"}
cox = [{"factor": lab[c], "hazard_ratio": s.loc[c, "exp(coef)"], "ci_low": s.loc[c, "exp(coef) lower 95%"], "ci_high": s.loc[c, "exp(coef) upper 95%"], "p": s.loc[c, "p"],
        "effect_on_speed_pct": 100 * (s.loc[c, "exp(coef)"] - 1)} for c in cols]
save("cox_model", cox, f"Cox proportional hazards on order completion. Hazard ratio > 1 = finishes faster. Concordance {cph.concordance_index_:.2f}.")
kpi("cox_c", cph.concordance_index_, "Cox model concordance")

# ---------------------------------------------------------------- throughput forecast + change points
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ruptures as rpt
last_full = (END - pd.Timedelta(days=END.dayofweek + 1)).normalize()
fc_rows, cp_rows = [], []
for key, sel, nm in (("despatch", d.work_centre.isin(["DESP", "MDESP"]), "Despatched units"), ("moulded", d.op.between(10, 19), "Moulded units")):
    w = d[sel & (d.date <= last_full)].set_index("date").qty.resample("W-SUN").sum()
    w = w[w.index >= "2021-01-10"]
    train, test = w[:-13], w[-13:]
    def fit(y): return ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
    bt = fit(train).forecast(13)
    mape = float(np.mean(np.abs(bt.values - test.values) / np.maximum(test.values, 1))) * 100
    naive = float(np.mean(np.abs(train.iloc[-13:].mean() - test.values) / np.maximum(test.values, 1))) * 100
    m = fit(w); f = m.forecast(13)
    res = (w - m.fittedvalues).std()
    for t, v in w[-104:].items(): fc_rows.append({"week": t.strftime("%Y-%m-%d"), "series": key, "actual": v, "forecast": None, "lo80": None, "hi80": None})
    for i, (t, v) in enumerate(f.items()):
        band = 1.2816 * res * math.sqrt(1 + i * 0.15)
        fc_rows.append({"week": t.strftime("%Y-%m-%d"), "series": key, "actual": None, "forecast": max(v, 0), "lo80": max(v - band, 0), "hi80": v + band})
    kpi(f"fc_{key}_wk", f.mean(), f"Forecast {nm.lower()} per week, next 13 weeks")
    kpi(f"fc_{key}_mape", mape, f"{nm} forecast back-test error (MAPE)", "%")
    kpi(f"fc_{key}_naive", naive, f"{nm} naive back-test error", "%")
    yr = d[sel].groupby(d.date.dt.year).qty.sum()
    kpi(f"{key}_2025", yr.get(2025, 0), f"{nm} 2025")
    kpi(f"{key}_2026_ann", yr.get(2026, 0) * 365 / END.dayofyear, f"{nm} 2026 annualised")
    sm = w.rolling(4, min_periods=1).mean().values.reshape(-1, 1)
    bk = rpt.Pelt(model="l2", min_size=12).fit(sm).predict(pen=np.var(sm) * 8)
    s0 = 0
    for b in bk:
        seg = w.iloc[s0:b]
        cp_rows.append({"series": key, "from": seg.index[0].strftime("%Y-%m-%d"), "to": seg.index[-1].strftime("%Y-%m-%d"), "mean_per_week": seg.mean(), "weeks": len(seg)})
        s0 = b
save("forecast", fc_rows, "Weekly units, Holt-Winters damped additive trend, 13-week horizon, approximate 80% band from residual spread.")
save("change_points", cp_rows, "Regime shifts in 4-week rolling weekly units, PELT change-point search (L2 cost).")
desp_cp = [r for r in cp_rows if r["series"] == "despatch"]
kpi("cp_desp_last", desp_cp[-1]["mean_per_week"], "Current despatch level per week")
kpi("cp_desp_prev", desp_cp[-2]["mean_per_week"] if len(desp_cp) > 1 else None, "Previous despatch level per week")
kpi("cp_desp_date", pd.Timestamp(desp_cp[-1]["from"]).strftime("%b %Y"), "Start of current despatch regime")

rec6 = d[d.date >= "2026-08-01"]; pre6 = d[(d.date >= "2026-01-01") & (d.date < "2026-08-01")]
m6, m0 = rec6.date.dt.to_period("M").nunique(), pre6.date.dt.to_period("M").nunique()
kpi("desp_pm_recent", rec6[rec6.work_centre.isin(["DESP", "MDESP"])].qty.sum() / m6, "Despatched units per month since Aug 2026")
kpi("desp_pm_prior", pre6[pre6.work_centre.isin(["DESP", "MDESP"])].qty.sum() / m0, "Despatched units per month Jan-Jul 2026")
kpi("fett_pm_recent", rec6[rec6.work_centre == "FETT"].qty.sum() / m6, "Fettled units per month since Aug 2026")
kpi("fett_pm_prior", pre6[pre6.work_centre == "FETT"].qty.sum() / m0, "Fettled units per month Jan-Jul 2026")
gap = d[(d.date >= "2026-08-01")]
kpi("unmatched_units", gap[gap.work_centre == "FETT"].qty.sum() - gap[gap.work_centre == "DESP"].qty.sum(), "Fettled but not booked to casting despatch since Aug 2026")
mon_fd = d[d.date >= "2025-01-01"].groupby([d.date.dt.to_period("M"), "work_centre"]).qty.sum().unstack().reindex(columns=["FETT", "DESP", "MDESP"]).fillna(0)
save("fettle_vs_despatch", [{"month": str(m), "fettled": r.FETT, "despatched_cast": r.DESP, "despatched_machined": r.MDESP} for m, r in mon_fd.iterrows()], "Units booked per month. 2026-11 is part month.")

# ---------------------------------------------------------------- supply chain analytics
dsp = d[d.work_centre.isin(["DESP", "MDESP"])].copy()
rec = dsp[dsp.date > END - pd.Timedelta(days=730)]
pq = rec.groupby("part_no").qty.sum().sort_values(ascending=False)
cum = pq.cumsum() / pq.sum()
abc = pd.Series(np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C")), index=pq.index)
mon = rec.groupby([rec.part_no, rec.date.dt.to_period("M")]).qty.sum().unstack(fill_value=0)
nz = mon.where(mon > 0)
adi = mon.shape[1] / (mon > 0).sum(axis=1)
cv2 = (nz.std(axis=1) / nz.mean(axis=1)).fillna(0) ** 2
xyz = pd.Series(np.where(adi < 1.32, np.where(cv2 < 0.49, "Smooth", "Erratic"), np.where(cv2 < 0.49, "Intermittent", "Lumpy")), index=mon.index)
ax = pd.DataFrame({"abc": abc, "xyz": xyz.reindex(abc.index), "qty": pq})
mat = ax.groupby(["abc", "xyz"]).agg(parts=("qty", "size"), qty=("qty", "sum")).reset_index()
mat["qty_share_pct"] = 100 * mat.qty / mat.qty.sum()
save("abc_xyz", mat, "Last 24 months despatched units. ABC: cumulative 80/95%. Demand pattern: Syntetos-Boylan, ADI cut-off 1.32 months, CV squared cut-off 0.49.")
kpi("a_parts", int((abc == "A").sum()), "A parts (80% of despatched units)")
kpi("parts_active_24m", len(pq), "Parts despatched in last 24 months")
kpi("a_parts_pct", 100 * (abc == "A").mean(), "Share of parts that are A", "%")
share = pq / pq.sum()
kpi("top10_share", 100 * share.head(10).sum(), "Top 10 parts share of despatched units", "%")
kpi("hhi", 10000 * (share ** 2).sum(), "Herfindahl-Hirschman index of part demand")
kpi("lumpy_pct", 100 * (xyz.reindex(abc.index) == "Lumpy").mean(), "Share of parts with lumpy demand", "%")
kpi("smooth_pct", 100 * (xyz.reindex(abc.index) == "Smooth").mean(), "Share of parts with smooth demand", "%")
kpi("intermittent_pct", 100 * (xyz.reindex(abc.index) == "Intermittent").mean(), "Share of parts with intermittent demand", "%")
pareto = [{"rank": i + 1, "part_no": p, "qty": q, "cum_pct": 100 * cum[p], "abc": abc[p], "xyz": xyz.get(p)} for i, (p, q) in enumerate(pq.head(25).items())]
save("pareto", pareto)
# portfolio churn
fy = d.groupby("part_no").date.agg(["min", "max"])
churn = []
for yr in range(2021, 2027):
    act = d[d.yr == yr].part_no.nunique()
    churn.append({"year": str(yr), "active_parts": act, "new_parts": int((fy["min"].dt.year == yr).sum()) if yr > 2021 else None,
                  "last_seen_parts": int((fy["max"].dt.year == yr).sum()) if yr < 2026 else None})
save("portfolio", churn, "Parts first booked and last booked each year.")
# despatch rhythm
rh = []
for yr in range(2021, 2027):
    g = dsp[dsp.yr == yr]
    rh.append({"year": str(yr), "last10_share_pct": 100 * g[g.date.dt.day > 21].qty.sum() / g.qty.sum(),
               "even_share_pct": 100 * 10 / 30.4})
save("despatch_rhythm", rh, "Share of despatched units in days 22 onward of each month vs an even spread.")
kpi("month_end_2025", rh[4]["last10_share_pct"], "Units despatched in final third of month 2025", "%")
wd = dsp.groupby(dsp.date.dt.dayofweek).qty.sum()
save("despatch_weekday", [{"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i], "qty": wd.get(i, 0)} for i in range(7)])
# order cadence for A parts
cad = []
for p in pq.index[(abc == "A").values][:15]:
    st_ = d[(d.part_no == p)].groupby("wo_base").date.min().sort_values()
    gaps = st_.diff().dt.days.dropna()
    q = d[(d.part_no == p)].groupby("wo_base").qty.max()
    cad.append({"part_no": p, "orders": len(st_), "median_gap_days": gaps.median() if len(gaps) else None,
                "batch_median": q.median(), "batch_cv": q.std() / q.mean() if q.mean() else None, "xyz": xyz.get(p)})
save("cadence", cad, "A parts: days between works orders and batch size variability.")
mat_lost = d[(d.type == "Lost") & d.lost_code.isin([7, 8])].hrs_clean.sum()
kpi("material_lost_hrs", mat_lost, "Hours lost to material availability", "hrs")
kpi("subcon_rows", int((d.work_centre == "SUBCON").sum()), "Subcontract bookings")

# ---------------------------------------------------------------- people coverage (anonymised)
rc = d[d.date > END - pd.Timedelta(days=730)]
cov_rows = []
for wc, g in rc.groupby("work_centre"):
    c = g.name.value_counts()
    if c.sum() < 100: continue
    q = c[c >= 20]
    cov_rows.append({"work_centre": wc, "entries": int(c.sum()), "operators_20plus": len(q), "top_operator_share_pct": 100 * c.iloc[0] / c.sum(),
                     "risk": "High" if (len(q) <= 1 or c.iloc[0] / c.sum() > 0.8) else ("Medium" if c.iloc[0] / c.sum() > 0.6 else "Low")})
cov_rows = sorted(cov_rows, key=lambda r: -r["top_operator_share_pct"])
save("coverage", cov_rows, "Last 24 months. Operators with 20+ bookings at the centre. Names withheld.")
kpi("single_cover", sum(r["risk"] == "High" for r in cov_rows), "Work centres dependent on one operator")
kpi("centres_cov", len(cov_rows), "Work centres assessed for cover")
ops_n = rc.groupby("name").work_centre.nunique()
kpi("ops_multi", int((ops_n >= 3).sum()), "Operators active on 3+ centres")
kpi("ops_total", len(ops_n), "Operators active in last 24 months")

# ---------------------------------------------------------------- quote model (LightGBM quantiles + TreeSHAP)
import lightgbm as lgb
jq = jobs[jobs.yr <= 2025].copy()
jq["log_qty"] = np.log(jq.qty); jq["op"] = jq.wo_no.str.split("/").str[1].astype(float)
jq["repeats"] = jq.groupby("part_no").cumcount(); jq["month"] = jq.date.dt.month
for c in ("part_no", "work_centre"): jq[c] = jq[c].astype("category")
F = ["part_no", "work_centre", "log_qty", "op", "repeats", "month"]
tr, te = jq[jq.yr <= 2023], jq[jq.yr >= 2024]
preds = {}
for a in (0.1, 0.5, 0.9):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=10, random_state=0, deterministic=True, n_jobs=1, verbose=-1)
    m.fit(tr[F], np.log(tr.hrs)); preds[a] = np.exp(m.predict(te[F]))
    if a == 0.5:
        contrib = m.predict(te[F], pred_contrib=True)
        imp = np.abs(contrib[:, :-1]).mean(axis=0)
y = te.hrs.values
coverage = float(np.mean((y >= preds[0.1]) & (y <= preds[0.9]))) * 100
mae = float(np.mean(np.abs(preds[0.5] - y)))
within25 = float(np.mean(np.abs(preds[0.5] - y) <= 0.25 * y)) * 100
wd_ = float(np.median(preds[0.9] - preds[0.1]))
names = {"part_no": "Part number", "work_centre": "Work centre", "log_qty": "Batch size", "op": "Operation number", "repeats": "Times made before", "month": "Month"}
save("quote_importance", sorted([{"factor": names[f], "mean_abs_shap": float(v)} for f, v in zip(F, imp)], key=lambda r: -r["mean_abs_shap"]),
     "Mean absolute TreeSHAP contribution to log job hours, median model, 2024-2025 test jobs.")
qs = te.assign(p10=preds[0.1], p50=preds[0.5], p90=preds[0.9]).sort_values("date")
sample = qs[qs.part_no.astype(str) == str(pq.index[0])].tail(20)
if len(sample) < 8: sample = qs.tail(20)
save("quote_sample", [{"job": f"{r.wo_no}", "part_no": str(r.part_no), "qty": r.qty, "actual": r.hrs, "p10": r.p10, "p50": r.p50, "p90": r.p90} for r in sample.itertuples()])
kpi("q_cov", coverage, "Jobs landing inside the predicted low-high range", "%")
kpi("q_mae", mae, "Quote model mean error per job", "hrs")
kpi("q_w25", within25, "Jobs within 25% of the likely estimate", "%")
kpi("q_width", wd_, "Median width of quoted range", "hrs")
kpi("q_n_test", len(te), "Test jobs 2024-2025")

# ---------------------------------------------------------------- queueing capacity (Kingman VUT)
vut = []
base = mc[mc.yr == 2024]
jb24 = base.groupby(["wo_no", "work_centre"]).agg(hrs=("hrs_clean", "sum"), date=("date", "min"), setup=("hrs_clean", lambda s: s[base.loc[s.index, "type"] == "Setup"].sum())).reset_index()
weeks24 = 52.0
scen = [("Today", 1.0, 1.0, 1.0), ("Demand +20%", 1.2, 1.0, 1.0), ("Setup halved", 1.0, 0.5, 1.0), ("Demand +20% and setup halved", 1.2, 0.5, 1.0)]
for wc, g in jb24.groupby("work_centre"):
    if len(g) < 30: continue
    ia = g.sort_values("date").date.diff().dt.days.dropna(); ia = ia[ia >= 0] + 0.2
    ca2 = (ia.std() / ia.mean()) ** 2
    for nm, dm, sf, _ in scen:
        te_ = (g.hrs - g.setup * (1 - sf)); ce2 = (te_.std() / te_.mean()) ** 2
        lam = len(g) / weeks24 * dm; u = lam * te_.mean() / SHIFT_HRS_WK
        wq = ((ca2 + ce2) / 2) * (u / (1 - u)) * te_.mean() / 8 if u < 0.98 else None
        vut.append({"work_centre": wc, "scenario": nm, "utilisation_pct": 100 * u, "queue_days": wq, "jobs_per_week": lam, "ca2": ca2, "ce2": ce2})
save("capacity_vut", vut, f"Kingman VUT approximation, single server, {SHIFT_HRS_WK:.0f} h/week, 2024 recorded hours. Queue days = working days waiting.")
vt = pd.DataFrame(vut)
bn = vt[vt.scenario == "Today"].sort_values("utilisation_pct", ascending=False).iloc[0]
kpi("bottleneck", bn.work_centre, "Most loaded machining centre"); kpi("bottleneck_u", bn.utilisation_pct, "Its recorded utilisation", "%")

# ---------------------------------------------------------------- data quality summary
dq = []
for yr in range(2021, 2027):
    g = d[d.yr == yr]; m = g[g.op.between(100, 269)]
    dq.append({"year": str(yr), "machining_timed_pct": 100 * (m.hrs_clean > 0).mean(), "phantom_hrs": g.loc[g.flag_hrs_mismatch, "hrs"].sum(),
               "lost_coded_pct": 100 * g[g.type == "Lost"].lost_time.notna().mean() if (g.type == "Lost").any() else None})
save("data_quality", dq)
kpi("phantom_total", d.loc[d.flag_hrs_mismatch, "hrs"].sum(), "Phantom booked hours", "hrs")
kpi("phantom_share", 100 * d.loc[d.flag_hrs_mismatch, "hrs"].sum() / (d.loc[d.flag_hrs_mismatch, "hrs"].sum() + d.hrs_clean.sum()), "Phantom share of recorded hours", "%")
kpi("timed_2026", dq[-1]["machining_timed_pct"], "Machining entries with time 2026", "%")
kpi("timed_2025", dq[-2]["machining_timed_pct"], "Machining entries with time 2025", "%")

# ================================================================ CASTING AND WHOLE CHAIN
LINES = ["ASF", "ASC", "GSL", "AIRSET SPURLINE", "DIE", "RAPID"]
mo = d[d.op.between(10, 19) & d.work_centre.isin(LINES)].copy()
ann = 365 / END.dayofyear
ly = mo.groupby(["work_centre", "yr"]).qty.sum().unstack(fill_value=0)
rows_ly = []
for wc, r in ly.iterrows():
    row = {"line": wc}
    for y in range(2021, 2027): row[f"y{y}"] = float(r.get(y, 0)) * (ann if y == 2026 else 1)
    row["peak_year"] = str(int(r.idxmax())); row["change_vs_peak_pct"] = 100 * (row["y2026"] / max(r.max(), 1) - 1)
    rows_ly.append(row)
save("cast_line_year", rows_ly, "Moulded units per line per year. 2026 annualised from 19 Nov.")
tot = mo.groupby("yr").qty.sum()
kpi("moulded_peak", tot.max(), "Peak annual moulded units"); kpi("moulded_peak_year", str(int(tot.idxmax())), "Peak moulding year")
kpi("moulded_2026_ann2", tot.get(2026, 0) * ann, "Moulded units 2026 annualised")
kpi("moulded_vs_peak", 100 * (tot.get(2026, 0) * ann / tot.max() - 1), "2026 moulded units vs peak year", "%")
lm = mo.groupby([mo.date.dt.to_period("M"), "work_centre"]).qty.sum().unstack(fill_value=0)
save("cast_line_month", [{"month": str(m_), **{k: float(r.get(k, 0)) for k in LINES}} for m_, r in lm.iterrows()], "Moulded units per month by line.")

# working days = weekdays with any booking in the plant
wd_all = pd.Series(sorted(d[d.date.dt.dayofweek < 5].date.unique()))
cast_wo = d[d.flow == "casting"].groupby(["wo_base", "op"]).qty.sum().unstack()
first_line = mo.sort_values("date").groupby("wo_base").work_centre.first()
ycl = ycast.join(first_line.rename("line"), how="left")
oee_c = []
for wc in LINES:
    g = mo[mo.work_centre == wc]
    if len(g) < 30: continue
    for per, sel in (("2021-2025", g.yr <= 2025), ("2025", g.yr == 2025), ("2026", g.yr == 2026)):
        gg = g[sel]
        if gg.empty: continue
        wdays = wd_all[(wd_all.dt.year >= gg.yr.min()) & (wd_all.dt.year <= gg.yr.max())]
        daily = gg.groupby("date").qty.sum()
        A = len(daily) / max(len(wdays), 1)
        best = g[g.yr <= 2025].groupby("date").qty.sum().quantile(0.9)
        P = min(daily.mean() / best, 1.0) if best > 0 else np.nan
        yy = ycl[(ycl.line == wc)]
        yy = yy[yy.yr.between(gg.yr.min(), min(gg.yr.max(), 2025))] if per != "2026" else yy[yy.yr == 2026]
        Qr = yy.desp.clip(upper=yy["first"]).sum() / yy["first"].sum() if len(yy) > 10 else np.nan
        oee_c.append({"line": wc, "period": per, "active_days_pct": 100 * A, "units_per_active_day": daily.mean(), "best_day_p90": best,
                      "performance_pct": 100 * P, "yield_pct": 100 * Qr if Qr == Qr else None,
                      "oee_proxy_pct": 100 * A * P * Qr if Qr == Qr else None, "units": gg.qty.sum()})
save("cast_oee", oee_c, "Output-based OEE proxy per moulding line (no time is booked on casting). A = days the line booked output / plant working days. P = mean units per active day / 90th percentile day (2021-2025). Q = quantity yield of orders moulded on the line.")
c25 = [r for r in oee_c if r["period"] == "2021-2025" and r["oee_proxy_pct"] is not None]
if c25:
    b_ = max(c25, key=lambda r: r["oee_proxy_pct"]); w_ = min(c25, key=lambda r: r["oee_proxy_pct"])
    kpi("cast_oee_best_line", b_["line"], "Best moulding line (OEE proxy)"); kpi("cast_oee_best", b_["oee_proxy_pct"], "Best moulding line OEE proxy", "%")
    kpi("cast_oee_worst_line", w_["line"], "Weakest moulding line (OEE proxy)"); kpi("cast_oee_worst", w_["oee_proxy_pct"], "Weakest moulding line OEE proxy", "%")
    wsum = sum(r["units"] for r in c25)
    kpi("cast_oee_avg", sum(r["oee_proxy_pct"] * r["units"] for r in c25) / wsum, "Moulding OEE proxy, unit-weighted", "%")

# yield by line and worst parts
yl = []
for wc in LINES:
    g = ycl[ycl.line == wc]
    if len(g) > 20:
        yl.append({"line": wc, "orders": len(g), "units_started": g["first"].sum(), "units_lost": (g["first"] - g.desp.clip(upper=g["first"])).sum(),
                   "yield_pct": 100 * g.desp.clip(upper=g["first"]).sum() / g["first"].sum(), "orders_short_pct": 100 * (g.ratio < 1).mean()})
save("cast_yield_line", sorted(yl, key=lambda r: r["yield_pct"]), "Casting orders 2021-2026 by moulding line. Units lost = started minus despatched, per order.")
ycp = ycl.join(d.groupby("wo_base").part_no.first().rename("part_no"))
pp = ycp.groupby("part_no").apply(lambda g: pd.Series({"orders": len(g), "units_started": g["first"].sum(), "units_lost": (g["first"] - g.desp.clip(upper=g["first"])).sum(), "line": g.line.mode().iat[0] if g.line.notna().any() else None}))
pp = pp[pp.orders >= 5]; pp["yield_pct"] = 100 * (1 - pp.units_lost / pp.units_started)
save("cast_yield_part", pp.sort_values("units_lost", ascending=False).head(15).reset_index(), "Parts with 5+ casting orders, ranked by units lost between moulding and despatch.")
kpi("cast_units_lost", ycl["first"].sum() - ycl.desp.clip(upper=ycl["first"]).sum(), "Casting units started but not despatched", "units")
kpi("cast_lost_top15_share", 100 * pp.sort_values("units_lost", ascending=False).head(15).units_lost.sum() / max(pp.units_lost.sum(), 1), "Share of lost units in 15 parts", "%")

# melt reference prefix (meaning to confirm)
nt = mo.dropna(subset=["notes"]).copy()
nt["ref"] = nt.notes.str.strip().str.upper().str.extract(r"^(F1|F2|F|B|A|K|M|C)")[0]
nt.loc[nt.ref == "F", "ref"] = "F (no number)"
mref = nt.groupby("wo_base").ref.first()
yref = ycl.join(mref.rename("ref")).dropna(subset=["ref"])
mr = yref.groupby("ref").apply(lambda g: pd.Series({"orders": len(g), "yield_pct": 100 * g.desp.clip(upper=g["first"]).sum() / g["first"].sum(), "orders_short_pct": 100 * (g.ratio < 1).mean()})).reset_index()
mr["main_line"] = mr.ref.map(nt.groupby("ref").work_centre.agg(lambda s: s.mode().iat[0]))
save("cast_melt_ref", mr.sort_values("orders", ascending=False), "Yield grouped by the prefix of the moulding note (for example F1-, F2-, b, a, k). Prefix meaning not confirmed.")

# stage buffers (units waiting between stages, casting flow)
st_dates = ev[ev.flow == "casting"].pivot_table(index="wo_base", columns="stage", values="first", aggfunc="min")
st_qty = ev[ev.flow == "casting"].pivot_table(index="wo_base", columns="stage", values="qty", aggfunc="sum")
days_b = pd.date_range("2021-01-04", END, freq="D")
buf_rows = []
for a, b, nm in (("Mould and cast", "Fettle", "Moulded, waiting to fettle"), ("Fettle", "Casting despatch", "Fettled, waiting to despatch")):
    if a not in st_dates or b not in st_dates: continue
    sub = pd.DataFrame({"a": st_dates[a], "b": st_dates[b], "q": st_qty[a]}).dropna(subset=["a"])
    sub["b"] = sub["b"].fillna(END + pd.Timedelta(days=1)); sub["q"] = sub["q"].fillna(0)
    sub = sub[((sub.b - sub.a).dt.days <= 365) & (sub.b > sub.a)]
    delta = np.zeros(len(days_b) + 1)
    np.add.at(delta, days_b.searchsorted(sub.a.values), sub.q.values)
    np.add.at(delta, days_b.searchsorted(sub.b.values), -sub.q.values)
    s_ = pd.Series(np.cumsum(delta)[:-1], index=days_b)
    for t, v in s_.resample("W").mean().items(): buf_rows.append({"week": t.strftime("%Y-%m-%d"), "buffer": nm, "units": v})
    kpi("buf_" + a.split()[0].lower(), s_[END - pd.Timedelta(days=90):].mean(), f"Average units {nm.lower()}, last 90 days", "units")
    kpi("buf_" + a.split()[0].lower() + "_2025", s_["2025"].mean(), f"Average units {nm.lower()}, 2025", "units")
save("cast_buffers", buf_rows, "Units waiting between casting stages per week (works order quantity from first booking at one stage to first booking at the next; orders waiting over a year excluded).")

# waits by line
wl = []
for wc in LINES:
    ids = first_line[first_line == wc].index
    sd = st_dates.reindex(ids)
    if "Fettle" in sd and len(sd) > 20:
        w1 = (sd["Fettle"] - sd["Mould and cast"]).dt.days.dropna()
        w2 = (sd.get("Casting despatch") - sd["Fettle"]).dt.days.dropna() if "Casting despatch" in sd else pd.Series(dtype=float)
        wl.append({"line": wc, "orders": len(sd), "mould_to_fettle_median": w1.median(), "mould_to_fettle_p90": w1.quantile(.9), "fettle_to_desp_median": w2.median(), "fettle_to_desp_p90": w2.quantile(.9)})
save("cast_waits_line", wl, "Calendar days between first bookings at each stage, by moulding line.")

# fettling crew
fe = d[(d.work_centre == "FETT")].copy()
fe["crew"] = fe.notes.fillna("").str.strip().str.split(r"[\s,/]+").apply(lambda x: len([t for t in x if t.isalpha() and 1 < len(t) <= 3]))
fdaily = fe.groupby("date").qty.sum()
fc_ = fe[fe.crew > 0].groupby("crew").agg(bookings=("qty", "size"), units_per_booking=("qty", "median")).reset_index()
fc_ = fc_[fc_.bookings >= 30]
save("fettle_crew", fc_, "Fettling bookings by number of initials in the note (crew size, assumed). Median units per booking.")
fy = fe.groupby("yr").agg(units=("qty", "sum"), days=("date", "nunique")).reset_index()
fy["units_per_day"] = fy.units / fy.days; fy["year"] = fy.yr.astype(str)
save("fettle_year", fy.drop(columns="yr"), "Fettled units, days with bookings and units per day.")

# end-to-end flow map (2025)
fm = []
y25 = d[d.yr == 2025]
for stg, sel in (("Mould and cast", y25.op.between(10, 19)), ("Fettle", y25.op.between(20, 59)), ("Casting inspection", y25.op.between(60, 69)), ("Core machining", y25.op.between(70, 79)),
                 ("Casting despatch", y25.op.between(80, 99)), ("Machining", y25.op.between(100, 199)), ("Final inspection", y25.op.between(200, 279)), ("Machined despatch", y25.op.between(280, 299))):
    g = y25[sel]
    fm.append({"stage": stg, "units": g.qty.sum(), "orders": g.wo_base.nunique(), "parts": g.part_no.nunique(), "people": g.name.nunique(), "work_centres": g.work_centre.nunique()})
fmw = {(a, b): v for a, b, v in zip(dg["from"], dg["to"], dg["median_days"])}
for r in fm:
    pair = {"Mould and cast": ("Mould and cast", "Fettle"), "Fettle": ("Fettle", "Casting despatch"), "Machining": ("Machining op 1", "Machined despatch")}.get(r["stage"])
    r["median_wait_to_next_days"] = fmw.get(pair) if pair else None
fm = [r for r in fm if r["units"] > 0]
save("flow_map", fm, "2025 volumes by stage; median days to the next stage from process mining.")
kpi("stream_overlap", len(set(d[d.op.between(10, 19)].part_no) & set(d[d.op.between(100, 269)].part_no)), "Part numbers both cast and machined here")
kpi("cast_parts", d[d.op.between(10, 19)].part_no.nunique(), "Part numbers cast")
kpi("mach_parts", d[d.op.between(100, 269)].part_no.nunique(), "Part numbers machined")
pr = d[(d.type == "Lost") & d.lost_time.str.contains("Pattern", na=False)]
kpi("pattern_repair_hrs", pr.hrs_clean.sum(), "Hours lost to pattern, die, core box and fixture repair", "hrs")
kpi("pattern_repair_760", pr[pr.work_centre == "760"].hrs_clean.sum(), "Of which on 760", "hrs")
kpi("pattern_repair_huv", pr[pr.work_centre == "HUVENE LATHE"].hrs_clean.sum(), "Of which on Huvene lathe", "hrs")

# ================================================================ LABOUR TIME MODEL (attendance-based)
from scipy.optimize import nnls
AREAS = {"ASF": "Moulding ASF", "ASC": "Moulding ASC", "GSL": "Moulding GSL", "AIRSET SPURLINE": "Moulding Airset", "DIE": "Moulding DIE", "RAPID": "Moulding Rapid",
         "FETT": "Fettling", "DESP": "Casting despatch", "MDESP": "Machined despatch", "CORE MACHINING": "Core machining", "CINSP": "Casting inspection"}
lb = d[d.date.dt.dayofweek < 5].copy()
lb["day_hrs"] = lb.date.dt.dayofweek.map(DAY_HRS)
# hours already booked on timed (machining) work that day come off the person's day
timed = lb[lb.hrs_clean > 0].groupby(["date", "name"]).hrs_clean.sum().clip(upper=12)
ua = lb[lb.work_centre.isin(AREAS)]
cnt = ua.groupby(["date", "name", "work_centre"]).size().rename("n").reset_index()
tot_n = lb.groupby(["date", "name"]).size().rename("all_n")
cnt = cnt.join(tot_n, on=["date", "name"]).join(timed.rename("timed"), on=["date", "name"])
cnt["timed"] = cnt.timed.fillna(0)
cnt["day_hrs"] = cnt.date.dt.dayofweek.map(DAY_HRS)
untimed_n = lb[lb.hrs_clean.fillna(0) == 0].groupby(["date", "name"]).size().rename("untimed_n")
cnt = cnt.join(untimed_n, on=["date", "name"])
cnt["hrs"] = (cnt.day_hrs - cnt.timed).clip(lower=0) * cnt.n / cnt.untimed_n.clip(lower=1)
rate = d[d.rate_gbp_hr > 0].groupby("name").rate_gbp_hr.median()
cnt["gbp"] = cnt.hrs * cnt.name.map(rate).fillna(d[d.rate_gbp_hr > 0].rate_gbp_hr.median())
ah = cnt.groupby(["work_centre", "date"]).agg(hrs=("hrs", "sum"), gbp=("gbp", "sum"), people=("name", "nunique")).reset_index()
# fettling crews: initials in notes are recorded inconsistently between years, so labour is booker-based (a minimum)
aq = ua.groupby(["work_centre", "date"]).qty.sum().rename("qty").reset_index()
ah = ah.merge(aq, on=["work_centre", "date"], how="left")
ah["yr"] = ah.date.dt.year
ay = ah.groupby(["work_centre", "yr"]).agg(labour_hrs=("hrs", "sum"), labour_gbp=("gbp", "sum"), units=("qty", "sum"), days=("date", "nunique"), people_per_day=("people", "mean")).reset_index()
ay["hrs_per_unit"] = ay.labour_hrs / ay.units.replace(0, np.nan)
ay["gbp_per_unit"] = ay.labour_gbp / ay.units.replace(0, np.nan)
ay["units_per_labour_hr"] = ay.units / ay.labour_hrs.replace(0, np.nan)
ay["area"] = ay.work_centre.map(AREAS); ay["year"] = ay.yr.astype(str)
ay = ay[ay.units > 0]
save("area_labour_year", ay.drop(columns="yr"), "Estimated labour hours per area: each operator's day (8.5 h Mon-Thu, 6 h Fri) minus hours booked on timed work, split across the areas they booked that day in proportion to bookings. Fettling crews are often booked by one person, so fettling hours are a minimum. Rate = operator median GBP/hr.")
fdy = ay[ay.work_centre.isin(["ASF", "ASC", "GSL", "AIRSET SPURLINE", "DIE", "RAPID", "FETT", "DESP"])].groupby("yr")[["labour_hrs", "labour_gbp"]].sum()
mq = d[d.op.between(10, 19)].groupby("yr").qty.sum()
fdy["units"] = mq; fdy["hrs_per_casting"] = fdy.labour_hrs / fdy.units; fdy["gbp_per_casting"] = fdy.labour_gbp / fdy.units
save("foundry_labour_year", [{"year": str(y), **{k: float(v) for k, v in r.items()}} for y, r in fdy.iterrows()], "Foundry labour (moulding, fettling, casting despatch) per moulded casting.")
kpi("hpc_2021", fdy.loc[2021, "hrs_per_casting"], "Foundry labour hours per casting 2021", "hrs")
kpi("hpc_2025", fdy.loc[2025, "hrs_per_casting"], "Foundry labour hours per casting 2025", "hrs")
kpi("hpc_2026", fdy.loc[2026, "hrs_per_casting"], "Foundry labour hours per casting 2026", "hrs")
kpi("gpc_2025", fdy.loc[2025, "gbp_per_casting"], "Foundry labour cost per casting 2025", "GBP")
kpi("gpc_2026", fdy.loc[2026, "gbp_per_casting"], "Foundry labour cost per casting 2026", "GBP")
kpi("foundry_hrs_2025", fdy.loc[2025, "labour_hrs"], "Foundry labour hours 2025", "hrs")

# per-part time per area: NNLS on daily area hours = sum(q_p,d * t_p) + c
pt_rows, fit_rows = [], []
for wc in AREAS:
    g = ua[(ua.work_centre == wc) & (ua.yr <= 2026)]
    if g.date.nunique() < 60: continue
    hd = ah[ah.work_centre == wc].set_index("date").hrs
    bd = g.groupby("part_no").date.nunique()
    keep = list(bd[bd >= 15].index)
    g = g.assign(pc=np.where(g.part_no.isin(keep), g.part_no, "Other parts"))
    X = g.pivot_table(index="date", columns="pc", values="qty", aggfunc="sum", fill_value=0)
    X = X.reindex(hd.index).fillna(0)
    A = np.c_[X.values, np.ones(len(X))]
    coef, _ = nnls(A, hd.values, maxiter=5000)
    pred = A @ coef; r2 = 1 - ((hd.values - pred) ** 2).sum() / ((hd.values - hd.values.mean()) ** 2).sum()
    fit_rows.append({"area": AREAS[wc], "days": len(X), "parts_modelled": len(keep), "r2": r2, "fixed_hrs_per_day": coef[-1]})
    # proportional allocation per part
    gq = g.groupby(["date", "part_no"]).qty.sum().reset_index()
    gq = gq.join(gq.groupby("date").qty.sum().rename("dq"), on="date").join(hd.rename("dh"), on="date")
    gq["alloc"] = gq.dh * gq.qty / gq.dq.replace(0, np.nan)
    pa = gq.groupby("part_no").agg(units=("qty", "sum"), alloc_hrs=("alloc", "sum"))
    for pc, t in zip(X.columns, coef[:-1]):
        if pc == "Other parts": continue
        u = pa.loc[pc, "units"]
        pt_rows.append({"part_no": pc, "area": AREAS[wc], "units": u, "nnls_hrs_per_unit": t, "alloc_hrs_per_unit": pa.loc[pc, "alloc_hrs"] / u if u else None,
                        "gbp_per_unit": t * float(rate.median())})
save("area_fit", fit_rows, "Non-negative least squares fit per area: daily labour hours = sum of units x hours per unit by part + fixed hours per day.")
pt = pd.DataFrame(pt_rows)
save("part_area_time", pt.sort_values("units", ascending=False), "Estimated labour hours per unit by part and area. NNLS = model estimate; alloc = day hours shared by quantity.")
# route total for top cast parts
topc = d[d.op.between(10, 19)].groupby("part_no").qty.sum().nlargest(20).index
rt = []
for p in topc:
    sub = pt[pt.part_no == p]
    row = {"part_no": p, "units": float(d[(d.op.between(10, 19)) & (d.part_no == p)].qty.sum())}
    for nm, sel in (("mould", sub.area.str.startswith("Moulding")), ("fettle", sub.area == "Fettling"), ("despatch", sub.area == "Casting despatch")):
        row[nm + "_hrs"] = float(sub[sel].nnls_hrs_per_unit.sum()) if sel.any() else None
        row[nm + "_alloc"] = float(sub[sel].alloc_hrs_per_unit.sum()) if sel.any() else None
    row["total_hrs"] = sum(v for k, v in row.items() if k.endswith("_hrs") and v is not None)
    row["total_alloc"] = sum(v for k, v in row.items() if k.endswith("_alloc") and v is not None)
    row["gbp_per_unit"] = row["total_alloc"] * float(rate.median())
    rt.append(row)
save("part_route_time", rt, "Top 20 cast parts by units: estimated labour hours per casting at moulding, fettling and despatch (allocation method), with model estimate alongside.")

# ================================================================ CASTING SIZE AND MIX (no weights in data: size = standard labour content)
mq_ = ua[ua.work_centre.isin(LINES)].groupby(["date", "work_centre", "part_no"]).qty.sum().reset_index()
mq_ = mq_.join(mq_.groupby(["date", "work_centre"]).qty.sum().rename("dq"), on=["date", "work_centre"]).merge(ah[["work_centre", "date", "hrs"]], on=["work_centre", "date"], how="left")
mq_["alloc"] = mq_.hrs * mq_.qty / mq_.dq.replace(0, np.nan)
std = mq_.groupby("part_no").agg(u=("qty", "sum"), h=("alloc", "sum"), days=("date", "nunique"))
std["std_hpu"] = std.h / std.u.replace(0, np.nan)
std["units_per_day"] = std.u / std.days
std = std[std.u > 0].dropna(subset=["std_hpu"])
cuts = [0, 0.1, 0.4, 1e9]; labs = ["Small (under 0.1 h)", "Medium (0.1 to 0.4 h)", "Large (over 0.4 h)"]
std["size"] = pd.cut(std.std_hpu, cuts, labels=labs).astype(str)
mo2 = mo.join(std[["std_hpu", "size"]], on="part_no").dropna(subset=["std_hpu"])
mo2["std_hrs"] = mo2.qty * mo2.std_hpu
mould_lab = ay[ay.work_centre.isin(LINES)].groupby("yr").labour_hrs.sum()
sz = []
for y in range(2021, 2027):
    g = mo2[mo2.yr == y]; f = ann if y == 2026 else 1.0
    row = {"year": str(y), "castings": g.qty.sum() * f, "standard_hrs": g.std_hrs.sum() * f, "avg_size_hrs": g.std_hrs.sum() / g.qty.sum(),
           "actual_mould_hrs": float(mould_lab.get(y, np.nan)) * f, "efficiency_pct": 100 * g.std_hrs.sum() / float(mould_lab.get(y, np.nan))}
    for l in labs: row[l.split(" ")[0].lower() + "_units"] = g[g["size"] == l].qty.sum() * f
    sz.append(row)
b0 = sz[0]
for r in sz:
    r["castings_idx"] = 100 * r["castings"] / b0["castings"]; r["std_hrs_idx"] = 100 * r["standard_hrs"] / b0["standard_hrs"]; r["size_idx"] = 100 * r["avg_size_hrs"] / b0["avg_size_hrs"]
save("size_mix_year", sz, "Size proxy: each part's standard moulding labour hours per casting (all-years average from the attendance model), held fixed. Standard hours = castings x standard hours per casting, so it measures work content regardless of mix. Efficiency = standard hours / actual moulding labour hours. 2026 annualised.")
pk = max(sz, key=lambda r: r["castings"]); l26 = sz[-1]
kpi("size_peak_year", pk["year"], "Peak castings year")
kpi("size_units_chg", 100 * (l26["castings"] / pk["castings"] - 1), "Castings 2026 vs peak", "%")
kpi("size_stdhrs_chg", 100 * (l26["standard_hrs"] / pk["standard_hrs"] - 1), "Work content (standard hours) 2026 vs peak-castings year", "%")
kpi("size_avg_chg", 100 * (l26["avg_size_hrs"] / pk["avg_size_hrs"] - 1), "Average casting size 2026 vs peak-castings year", "%")
kpi("size_avg_chg_2021", 100 * (l26["avg_size_hrs"] / b0["avg_size_hrs"] - 1), "Average casting size 2026 vs 2021", "%")
kpi("eff_2025", sz[4]["efficiency_pct"], "Moulding efficiency 2025 (standard / actual hours)", "%")
kpi("eff_2026", l26["efficiency_pct"], "Moulding efficiency 2026", "%")
kpi("eff_best", max(r["efficiency_pct"] for r in sz), "Best year moulding efficiency", "%")
kpi("eff_best_year", max(sz, key=lambda r: r["efficiency_pct"])["year"], "Best efficiency year")
kpi("small_share_peak", 100 * pk["small_units"] / pk["castings"], "Small castings share in peak year", "%")
kpi("small_share_2026", 100 * l26["small_units"] / l26["castings"], "Small castings share 2026", "%")
sl = mo2.groupby(["work_centre", "size"]).qty.sum().unstack(fill_value=0)
save("size_by_line", [{"line": l, **{c.split(" ")[0].lower() + "_units": float(r.get(c, 0)) for c in labs}, "avg_size_hrs": float((mo2[mo2.work_centre == l].std_hrs.sum()) / max(mo2[mo2.work_centre == l].qty.sum(), 1))} for l, r in sl.iterrows()], "Castings by size class per moulding line, 2021-2026.")
ysz = ycl.join(d.groupby("wo_base").part_no.first().rename("part_no")).join(std[["size"]], on="part_no").dropna(subset=["size"])
save("size_yield", [{"size": l, "orders": int((ysz["size"] == l).sum()), "yield_pct": 100 * ysz[ysz["size"] == l].desp.clip(upper=ysz[ysz["size"] == l]["first"]).sum() / max(ysz[ysz["size"] == l]["first"].sum(), 1),
                     "parts": int((std["size"] == l).sum())} for l in labs], "Casting yield and part count by size class.")

# ---- 2026 beside the whole dataset (added tables, prefix y26_)
wo_start = d.groupby("wo_base").date.min()
def _in26(idx): return wo_start.reindex(idx).dt.year.values == 2026
def _yl(g):
    f_ = g["first"].sum(); k_ = g.desp.clip(upper=g["first"]).sum()
    return {"orders": len(g), "started": f_, "lost": f_ - k_, "yield_pct": 100 * k_ / f_ if f_ else None}
def _both(name, a, b): return {"group": name, **{k + "_all": v for k, v in a.items()}, **{k + "_2026": v for k, v in b.items()}}
def _injj(idx): return _in26(idx) & (wo_start.reindex(idx).values < np.datetime64("2026-08-01"))
save("y26_yield_line", [{**_both(wc, _yl(g), _yl(g[_in26(g.index)])), "orders_janjul": int(_injj(g.index).sum()), "yield_pct_janjul": _yl(g[_injj(g.index)])["yield_pct"]} for wc in LINES + ["All lines"] for g in [ycl if wc == "All lines" else ycl[ycl.line == wc]]],
     "Casting yield by moulding line: all orders 2021-2026 beside orders started in 2026. Recent 2026 orders may not be despatched yet.")
_pa = ycp.groupby("part_no").apply(lambda g: pd.Series(_yl(g))); _p26 = ycp[_in26(ycp.index)].groupby("part_no").apply(lambda g: pd.Series(_yl(g)))
_pl = ycp.groupby("part_no").line.agg(lambda x: x.mode().iat[0] if x.notna().any() else None)
_e = {"orders": 0, "started": 0, "lost": 0, "yield_pct": None}
def _prow(p): return {**_both(p, _pa.loc[p].to_dict(), _p26.loc[p].to_dict() if p in _p26.index else _e), "line": _pl.get(p)}
save("y26_yield_part_all", [_prow(p) for p in _pa[_pa.orders >= 5].sort_values("lost", ascending=False).head(10).index], "Ten parts losing the most castings 2021-2026, with their 2026 figures.")
save("y26_yield_part_2026", [_prow(p) for p in _p26.sort_values("lost", ascending=False).head(10).index] if len(_p26) else [], "Ten parts losing the most castings on orders started in 2026, with their all-years figures.")
def _w(sd):
    w1 = (sd["Fettle"] - sd["Mould and cast"]).dt.days.dropna(); w2 = (sd["Casting despatch"] - sd["Fettle"]).dt.days.dropna()
    return {"orders": len(sd), "mf_median": w1.median(), "mf_p90": w1.quantile(.9), "fd_median": w2.median(), "fd_p90": w2.quantile(.9)}
_wr = []
for wc in LINES + ["All lines"]:
    sd = st_dates if wc == "All lines" else st_dates.reindex(first_line[first_line == wc].index)
    sd = sd.dropna(subset=["Mould and cast"])
    _wr.append(_both(wc, _w(sd), _w(sd[sd["Mould and cast"].dt.year == 2026])))
save("y26_waits_line", _wr, "Days between first bookings at each stage, all orders beside orders moulded in 2026. 2026 orders still waiting are not counted, so 2026 waits are understated.")
_fr = []
for stg, lo, hi in (("Mould and cast", 10, 19), ("Fettle", 20, 59), ("Casting inspection", 60, 69), ("Core machining", 70, 79), ("Casting despatch", 80, 99), ("Machining", 100, 199), ("Final inspection", 200, 279), ("Machined despatch", 280, 299)):
    def _f(g): return {"units": g.qty.sum(), "orders": g.wo_base.nunique(), "parts": g.part_no.nunique(), "people": g.name.nunique()}
    g = d[d.op.between(lo, hi)]
    if g.qty.sum() > 0: _fr.append(_both(stg, _f(g), _f(g[g.yr == 2026])))
save("y26_flow", _fr, "Bookings by stage, whole dataset (4 Jan 2021 to 19 Nov 2026) beside 2026 only.")
_m26 = mo2[mo2.date.dt.year == 2026]
def _s(g): return {**{c.split(" ")[0].lower(): float(g[g["size"] == c].qty.sum()) for c in labs}, "avg_size": float(g.std_hrs.sum() / g.qty.sum()) if g.qty.sum() else None}
save("y26_size_line", [_both(l, _s(g), _s(g[g.date.dt.year == 2026])) for l in LINES + ["All lines"] for g in [mo2 if l == "All lines" else mo2[mo2.work_centre == l]]], "Castings by size class per moulding line, whole dataset beside 2026.")
save("y26_size_yield", [_both(l, _yl(ysz[ysz["size"] == l]), _yl(ysz[(ysz["size"] == l) & _in26(ysz.index)])) for l in labs], "Casting yield by size class, all orders beside orders started in 2026.")

save("kpis", KPI)
print("report done", flush=True)
