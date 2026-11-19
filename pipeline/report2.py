"""PCT addendum analytics: OEE per area, commercial cross-reference, correlations.

Usage: python report2.py <wip2.parquet> <stages.csv> <report_dir> <pipeline_dir>
Adds JSON files to report_dir (prefix x_). Does not change report.py outputs.
"""
import sys, os, json, math, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
SRC, STAGES, OUT, PIPE = sys.argv[1:5]
CODE = open(os.path.abspath(__file__)).read()
DAY_HRS = {0: 8.5, 1: 8.5, 2: 8.5, 3: 8.5, 4: 6.0}
BENCH = {"availability": 90.0, "performance": 95.0, "quality": 99.0, "oee": 85.0, "typical_low": 55.0, "typical_high": 65.0}
PENSION = 3.0

def clean(v):
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)): return bool(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [{"label": "report2.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]
    json.dump({"rows": rows, "meta": meta}, open(f"{OUT}/x_{name}.json", "w"), allow_nan=False); print("wrote x_" + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]

d = pd.read_parquet(SRC); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); END = d.date.max()
LINES = ["ASF", "ASC", "GSL", "AIRSET SPURLINE", "DIE"]
AREAS = {"ASF": "Moulding ASF", "ASC": "Moulding ASC", "GSL": "Moulding GSL", "AIRSET SPURLINE": "Moulding Airset", "DIE": "Moulding DIE", "FETT": "Fettling", "DESP": "Casting despatch", "MDESP": "Machined despatch"}

# ---- attendance model (same rules as report.py)
lb = d[d.date.dt.dayofweek < 5].copy()
timed = lb[lb.hrs_clean > 0].groupby(["date", "name"]).hrs_clean.sum().clip(upper=12)
ua = lb[lb.work_centre.isin(AREAS)]
cnt = ua.groupby(["date", "name", "work_centre"]).size().rename("n").reset_index()
cnt = cnt.join(timed.rename("timed"), on=["date", "name"]).join(lb[lb.hrs_clean.fillna(0) == 0].groupby(["date", "name"]).size().rename("un"), on=["date", "name"])
cnt["hrs"] = (cnt.date.dt.dayofweek.map(DAY_HRS) - cnt.timed.fillna(0)).clip(lower=0) * cnt.n / cnt.un.clip(lower=1)
ah = cnt.groupby(["work_centre", "date"]).hrs.sum().rename("hrs").reset_index()
pq = ua.groupby(["work_centre", "date", "part_no"]).qty.sum().reset_index()
pq = pq.join(pq.groupby(["work_centre", "date"]).qty.sum().rename("dq"), on=["work_centre", "date"]).merge(ah, on=["work_centre", "date"], how="left")
pq["alloc"] = pq.hrs * pq.qty / pq.dq.replace(0, np.nan)
std = pq.groupby(["work_centre", "part_no"]).agg(u=("qty", "sum"), h=("alloc", "sum"))
std["t"] = std.h / std.u.replace(0, np.nan)
pq = pq.join(std.t, on=["work_centre", "part_no"]); pq["earned"] = pq.qty * pq.t
day = pq.groupby(["work_centre", "date"]).agg(earned=("earned", "sum"), units=("qty", "sum"), hrs=("hrs", "first")).reset_index(); day["yr"] = day.date.dt.year
wd = pd.Series(sorted(d[d.date.dt.dayofweek < 5].date.unique())); wdy = wd.groupby(wd.dt.year).size()

# ---- stage yields per order
q = d[d.op.notna()].groupby(["wo_base", "op"]).qty.sum().unstack()
start = d.groupby("wo_base").date.min(); mature = start < END - pd.Timedelta(days=45)
first_wc = d[d.op.between(10, 19)].sort_values("date").groupby("wo_base").work_centre.first()
def col(lo, hi): 
    c = [x for x in q.columns if lo <= x <= hi]; return q[c].sum(axis=1, min_count=1)
m_, f_, dsp, mm_, md = col(10, 19), col(20, 59), col(80, 99), col(100, 199), col(280, 299)
def yld(a, b, idx, yr=None):
    x = pd.DataFrame({"a": a, "b": b}).dropna(); x = x[(x.a > 0) & mature.reindex(x.index).fillna(False)]
    if idx is not None: x = x[x.index.isin(idx)]
    if yr is not None: x = x[start.reindex(x.index).dt.year == yr]
    return (x.b.clip(upper=x.a).sum() / x.a.sum(), len(x)) if len(x) >= 15 else (np.nan, len(x))

rows = []
for wc, nm in AREAS.items():
    g = day[day.work_centre == wc]
    if g.date.nunique() < 60: continue
    best = g[g.yr <= 2025].earned.quantile(0.9)
    for per in ["2021", "2022", "2023", "2024", "2025", "2026", "2021-2025", "2021-2026"]:
        gg = g[g.yr <= 2025] if per == "2021-2025" else g if per == "2021-2026" else g[g.yr == int(per)]
        if gg.empty: continue
        yrs = range(2021, 2026) if per == "2021-2025" else range(2021, 2027) if per == "2021-2026" else [int(per)]
        A = len(gg) / sum(wdy.get(y, 0) for y in yrs)
        P = min(gg.earned.mean() / best, 1.0)
        yr = None if "-" in per else int(per)
        if wc in LINES: Q, n = yld(m_, f_, first_wc[first_wc == wc].index, yr); qb = "fettled / moulded"
        elif wc == "FETT": Q, n = yld(f_, dsp, None, yr); qb = "despatched / fettled"
        else: Q, n = 1.0, 0; qb = "no downstream stage; set to 100%"
        rows.append({"area": nm, "period": per, "availability": 100 * A, "performance": 100 * P, "quality": 100 * Q if Q == Q else None, "oee": 100 * A * P * Q if Q == Q else None,
                     "labour_efficiency": 100 * gg.earned.sum() / gg.hrs.sum(), "std_hrs_per_day": gg.earned.mean(), "best_day_p90": best, "producing_days": len(gg), "quality_basis": qb, "method": "output-based"})
for r in J("oee_year"): rows.append({"area": "Machining (all centres)", "period": r["year"], "availability": r["availability"], "performance": r["performance"], "quality": r["quality"], "oee": r["oee"], "labour_efficiency": None, "quality_basis": "despatched / started", "method": "time-based"})
K = {r["key"]: r["value"] for r in J("kpis")}
rows.append({"area": "Machining (all centres)", "period": "2021-2025", "availability": K["oee_availability"], "performance": K["oee_performance"], "quality": K["oee_quality"], "oee": K["oee_oee"], "labour_efficiency": None, "quality_basis": "despatched / started", "method": "time-based"})
for r in rows:
    for k in ("availability", "performance", "quality", "oee"): r[k + "_gap"] = None if r.get(k) is None else r[k] - BENCH[k]
save("area_oee", rows, "OEE per area. Foundry areas book no time, so A = producing days / plant working days; P = mean standard hours earned per producing day / 90th percentile day (2021-2025), which removes casting size from the rate; Q = stage yield to the next stage. Machining is time-based. Gaps are against Nakajima's world-class components (A 90, P 95, Q 99, OEE 85).")
save("oee_benchmarks", [{"measure": "Availability", "world_class": 90, "source": "Nakajima (1988)"}, {"measure": "Performance", "world_class": 95, "source": "Nakajima (1988)"}, {"measure": "Quality", "world_class": 99, "source": "Nakajima (1988)"}, {"measure": "OEE", "world_class": 85, "source": "Nakajima (1988)"},
                        {"measure": "OEE, cross-industry observed", "world_class": 55, "source": "Ljungberg (1998), about 20 cases"}, {"measure": "OEE, cross-industry observed", "world_class": 65, "source": "Hedman et al. (2016), 884 machines"}])

# ---- commercial
ext = pd.read_csv(f"{PIPE}/external.csv").set_index("year")
fl = {int(r["year"]): r for r in J("foundry_labour_year")}; sz = {int(r["year"]): r for r in J("size_mix_year")}
ann = 365 / END.dayofyear; com = []
med_rate = d[d.rate_gbp_hr > 0].groupby("yr").rate_gbp_hr.median()
for y in range(2021, 2027):
    nlw = 0.25 * ext.loc[y - 1, "nlw_gbp_hr_apr"] + 0.75 * ext.loc[y, "nlw_gbp_hr_apr"]
    ni = ext.loc[y, "employer_ni_pct"] if y != 2025 else 0.25 * 13.8 + 0.75 * 15.0
    loaded = nlw * (1 + (ni + PENSION) / 100)
    hrs, units = fl[y]["labour_hrs"], fl[y]["units"]
    stdh = sz[y]["standard_hrs"] / (ann if y == 2026 else 1)
    kg = ext.loc[y, "lme_usd_t"] / ext.loc[y, "gbp_usd_approx"] / 1000
    com.append({"year": str(y), "foundry_labour_hrs": hrs, "castings": units, "booked_rate_median": float(med_rate.get(y, np.nan)), "nlw_blended": nlw, "loaded_min_rate": loaded,
                "booked_rate_vs_nlw_pct": 100 * (float(med_rate.get(y, np.nan)) / nlw - 1),
                "labour_gbp_booked": fl[y]["labour_gbp"], "labour_gbp_loaded_min": hrs * loaded, "gbp_per_casting_booked": fl[y]["gbp_per_casting"], "gbp_per_casting_loaded_min": hrs * loaded / units,
                "labour_hrs_per_std_hr": hrs / stdh, "gbp_per_std_hr_loaded_min": hrs * loaded / stdh,
                "lme_usd_t": ext.loc[y, "lme_usd_t"], "lme_basis": ext.loc[y, "lme_basis"], "al_gbp_per_kg": kg, "breakeven_kg": hrs * loaded / units / kg})
b = com[0]
for r in com:
    r["idx_lme"] = 100 * r["lme_usd_t"] / b["lme_usd_t"]; r["idx_nlw"] = 100 * r["nlw_blended"] / b["nlw_blended"]; r["idx_loaded_rate"] = 100 * r["loaded_min_rate"] / b["loaded_min_rate"]
    r["idx_labour_per_std_hr"] = 100 * r["gbp_per_std_hr_loaded_min"] / b["gbp_per_std_hr_loaded_min"]; r["idx_cost_per_casting"] = 100 * r["gbp_per_casting_loaded_min"] / b["gbp_per_casting_loaded_min"]
save("commercial_year", com, "Labour: foundry attendance hours x statutory minimum loaded rate (National Living Wage blended Jan-Mar/Apr-Dec, plus employer NI at the headline rate and 3% pension; thresholds, holiday pay and overtime ignored, so this is a floor). Aluminium: LME annual average USD/t (2026 is a forecast), converted at approximate annual GBP/USD. Break-even kg = casting weight at which LME metal value equals minimum labour cost per casting; ignores melt loss, alloy premium and returns.")
sens = [{"casting_kg": w, **{f"metal_gbp_{r['year']}": w * r["al_gbp_per_kg"] for r in com}, "labour_gbp_2026": com[-1]["gbp_per_casting_loaded_min"], "metal_share_2026_pct": 100 * w * com[-1]["al_gbp_per_kg"] / (w * com[-1]["al_gbp_per_kg"] + com[-1]["gbp_per_casting_loaded_min"])} for w in (0.5, 1, 2, 5, 10, 20)]
save("metal_sensitivity", sens, "LME metal value per casting at assumed weights, against 2026 average minimum labour cost per casting. Weights are assumptions: no weights in the data.")
rr = d[d.rate_gbp_hr > 0].groupby(["yr", "rate_gbp_hr"]).size().rename("entries").reset_index()
rr = rr[rr.groupby("yr").entries.transform(lambda s: s / s.sum()) >= 0.03]
save("booked_rates", [{"year": str(int(r.yr)), "rate_gbp_hr": r.rate_gbp_hr, "entries": int(r.entries)} for r in rr.itertuples()], "Hourly rates recorded on bookings, rates with 3%+ of a year's rated entries.")

# ---- correlations: monthly
mth = lambda s: s.dt.to_period("M")
mo = d[d.op.between(10, 19) & d.work_centre.isin(LINES)]
mstd = pq[pq.work_centre.isin(LINES)].groupby(mth(pq[pq.work_centre.isin(LINES)].date)).earned.sum()
M = pd.DataFrame({"Castings moulded": mo.groupby(mth(mo.date)).qty.sum(), "Moulding work content (std hrs)": mstd})
M["Average casting size"] = M.iloc[:, 1] / M.iloc[:, 0]
M["Castings fettled"] = d[d.work_centre == "FETT"].groupby(mth(d[d.work_centre == "FETT"].date)).qty.sum()
M["Castings despatched"] = d[d.work_centre == "DESP"].groupby(mth(d[d.work_centre == "DESP"].date)).qty.sum()
M["Machined units despatched"] = d[d.work_centre == "MDESP"].groupby(mth(d[d.work_centre == "MDESP"].date)).qty.sum()
M["Foundry labour hours"] = ah[ah.work_centre.isin(LINES + ["FETT", "DESP"])].groupby(mth(ah[ah.work_centre.isin(LINES + ["FETT", "DESP"])].date)).hrs.sum()
M["Distinct parts moulded"] = mo.groupby(mth(mo.date)).part_no.nunique()
M["People booking in foundry"] = d[d.work_centre.isin(LINES + ["FETT"])].groupby(mth(d[d.work_centre.isin(LINES + ["FETT"])].date)).name.nunique()
sd = d[d.stage_order.notna()] if "stage_order" in d else None
fs = d[d.op.between(10, 19)].groupby("wo_base").date.min(); ff = d[d.op.between(20, 59)].groupby("wo_base").date.min(); fd_ = d[d.op.between(80, 99)].groupby("wo_base").date.min()
o = pd.DataFrame({"mould": fs, "fettle": ff, "desp": fd_}).dropna(subset=["mould"])
o["wait_mf"] = (o.fettle - o.mould).dt.days; o["lead"] = (o.desp - o.mould).dt.days
o["qty"] = m_.reindex(o.index); o["yield"] = (dsp.reindex(o.index).clip(upper=o.qty) / o.qty)
part = d.groupby("wo_base").part_no.first(); o["line"] = first_wc.reindex(o.index)
o["size"] = [std.t.get((l, p), np.nan) for l, p in zip(o.line, part.reindex(o.index))]
o["repeat"] = part.reindex(o.index).map(d.groupby("part_no").wo_base.nunique()); o["month_end_start"] = (o.mould.dt.day > 21).astype(int)
om = o[(o.wait_mf >= 0) & (o.wait_mf <= 365)]
M["Mould to fettle wait (median days)"] = om.groupby(mth(om.mould)).wait_mf.median()
ol = o[(o.lead >= 0) & (o.lead <= 365)]; M["Casting lead time (median days)"] = ol.groupby(mth(ol.mould)).lead.median()
oy = o[mature.reindex(o.index).fillna(False) & o["yield"].notna()]; M["Casting yield"] = oy.groupby(mth(oy.mould)).apply(lambda g: (g["yield"] * g.qty).sum() / g.qty.sum())
mc = d[(d.hrs_clean > 0) & (d.op >= 100)]; M["Machining hours booked"] = mc.groupby(mth(mc.date)).hrs_clean.sum()
M["Lost time hours"] = d[(d.type == "Lost")].groupby(mth(d[d.type == "Lost"].date)).hrs_clean.sum()
M = M[(M.index >= "2021-01") & (M.index < END.to_period("M"))]
TIMEVARS = ["Machining hours booked", "Lost time hours"]
def corr_pairs(F, kind, min_n=24):
    out = []; cols = list(F.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = F[[cols[i], cols[j]]].dropna()
            if cols[i] in TIMEVARS or cols[j] in TIMEVARS: x = x[x.index < "2026-01"]
            if len(x) < min_n or x.iloc[:, 0].nunique() < 5 or x.iloc[:, 1].nunique() < 5: continue
            r, p = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1]); out.append({"a": cols[i], "b": cols[j], "rho": r, "p": p, "n": len(x), "kind": kind})
    if out:
        ps = np.array([o_["p"] for o_ in out]); order = np.argsort(ps); adj = np.empty(len(ps)); prev = 1.0
        for rank, idx in list(enumerate(order, 1))[::-1]: prev = min(prev, ps[idx] * len(ps) / rank); adj[idx] = prev
        for o_, a in zip(out, adj): o_["p_fdr"] = a; o_["significant"] = bool(a < 0.05)
    return out
lev = corr_pairs(M, "level"); dif = corr_pairs(M.diff(), "month-on-month change")
dd = {(r["a"], r["b"]): r for r in dif}
for r in lev:
    x = dd.get((r["a"], r["b"])); r["rho_change"] = x["rho"] if x else None; r["sig_change"] = x["significant"] if x else None
    r["robust"] = bool(r["significant"] and x and x["significant"] and np.sign(x["rho"]) == np.sign(r["rho"]))
save("corr_monthly", sorted(lev, key=lambda r: -abs(r["rho"])), f"Spearman rank correlation on {len(M)} monthly observations, Benjamini-Hochberg false discovery rate 5%. 'robust' = also significant with the same sign on month-on-month changes, which removes shared trends. Time-booked series stop at Dec 2025.")
cols = list(M.columns); cm = M.corr(method="spearman")
save("corr_matrix", [{"a": a, "b": b_, "rho": cm.loc[a, b_]} for a in cols for b_ in cols], "Spearman matrix of monthly series (levels).")
save("monthly_series", [{"month": str(i), **{c: r[c] for c in cols}} for i, r in M.iterrows()])
# order level
O = pd.DataFrame({"Batch size": o.qty, "Casting size (std hrs)": o["size"], "Mould to fettle wait": o.wait_mf.where((o.wait_mf >= 0) & (o.wait_mf <= 365)), "Lead time": o.lead.where((o.lead >= 0) & (o.lead <= 365)),
                  "Yield": o["yield"].where(mature.reindex(o.index).fillna(False)), "Times part ordered": o.repeat, "Started after 21st": o.month_end_start, "Start year": o.mould.dt.year})
save("corr_orders", sorted(corr_pairs(O, "order", 200), key=lambda r: -abs(r["rho"])), f"Spearman correlation across {len(O)} casting works orders, Benjamini-Hochberg 5%.")
O26 = O[O["Start year"] == 2026].drop(columns="Start year"); c26 = {(r["a"], r["b"]): r for r in corr_pairs(O26, "order 2026", 60)}
save("corr_orders_2026", [{"a": r["a"], "b": r["b"], "rho_all": r["rho"], "n_all": r["n"], "rho_2026": c26.get((r["a"], r["b"]), {}).get("rho"), "n_2026": c26.get((r["a"], r["b"]), {}).get("n"), "sig_2026": c26.get((r["a"], r["b"]), {}).get("significant")} for r in sorted(corr_pairs(O.drop(columns="Start year"), "order", 200), key=lambda r: -abs(r["rho"]))], "Order-level Spearman correlations, all orders beside orders started in 2026.")
M26 = M[M.index >= "2026-01"]; m26 = {(r["a"], r["b"]): r for r in corr_pairs(M26, "2026", 8)}
save("corr_monthly_2026", [{"a": r["a"], "b": r["b"], "rho_all": r["rho"], "robust_all": r["robust"], "rho_2026": m26.get((r["a"], r["b"]), {}).get("rho"), "p_2026": m26.get((r["a"], r["b"]), {}).get("p"), "n_2026": m26.get((r["a"], r["b"]), {}).get("n")} for r in sorted(lev, key=lambda r: -abs(r["rho"]))], f"Monthly Spearman correlations, 70 months beside the {len(M26)} complete months of 2026. Ten points is too few to test; 2026 values are indicative only.")
# lag: weekly moulded -> fettled / despatched
W = pd.DataFrame({"m": mo.set_index("date").qty.resample("W").sum(), "f": d[d.work_centre == "FETT"].set_index("date").qty.resample("W").sum(), "d": d[d.work_centre == "DESP"].set_index("date").qty.resample("W").sum()}).fillna(0)
W = W[(W.index >= "2021-01-10") & (W.index < "2026-08-01")]
lag = []
for k in range(0, 9):
    lag.append({"lag_weeks": k, "moulded_to_fettled": stats.spearmanr(W.m.iloc[:len(W) - k] if k else W.m, W.f.shift(-k).dropna())[0], "fettled_to_despatched": stats.spearmanr(W.f.iloc[:len(W) - k] if k else W.f, W.d.shift(-k).dropna())[0]})
save("lag_weekly", lag, "Spearman correlation of weekly units with the next stage k weeks later, Jan 2021 to Jul 2026.")
# aluminium quarterly
aq = pd.read_csv(f"{PIPE}/aluminium_quarterly.csv"); aq.index = pd.PeriodIndex(aq.quarter, freq="Q")
Qf = pd.DataFrame({"al": aq.usd_t, "Castings moulded": mo.groupby(mo.date.dt.to_period("Q")).qty.sum(), "Moulding work content (std hrs)": pq[pq.work_centre.isin(LINES)].groupby(pq[pq.work_centre.isin(LINES)].date.dt.to_period("Q")).earned.sum(),
                   "Castings despatched": d[d.work_centre == "DESP"].groupby(d[d.work_centre == "DESP"].date.dt.to_period("Q")).qty.sum(), "Machined units despatched": d[d.work_centre == "MDESP"].groupby(d[d.work_centre == "MDESP"].date.dt.to_period("Q")).qty.sum(),
                   "New works orders": start.groupby(start.dt.to_period("Q")).size()}).dropna()
alc = []
for c in Qf.columns[1:]:
    for k in (0, 1, 2):
        x = pd.DataFrame({"a": Qf.al, "b": Qf[c].shift(-k)}).dropna(); r, p = stats.spearmanr(x.a, x.b)
        xd = x.diff().dropna(); r2, p2 = stats.spearmanr(xd.a, xd.b)
        alc.append({"series": c, "lag_quarters": k, "rho": r, "p": p, "rho_change": r2, "p_change": p2, "n": len(x)})
save("corr_aluminium", alc, "Quarterly LME aluminium price (IMF series via FRED, 2021Q1 to 2025Q3) against PCT volumes, same quarter and one or two quarters later. n is small; treat as indicative.")
save("aluminium_quarterly", [{"quarter": str(i), "usd_t": r.al, "castings": r["Castings moulded"], "std_hrs": r["Moulding work content (std hrs)"]} for i, r in Qf.iterrows()])
print("report2 done", flush=True)
