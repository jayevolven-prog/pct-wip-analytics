"""PCT extended models on the loaded bookings: repricing, churn, learning, prediction, control charts, capacity, calendar, variety, notes, booking discipline, operators.

Usage: python report3.py <wip2.parquet> <stages.csv> <report_dir> <pipeline_dir>
Writes JSON files with prefix z_. Operator names are replaced by codes.
"""
import sys, os, json, math, re, warnings
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SRC, STAGES, OUT, PIPE = sys.argv[1:5]
CODE = open(os.path.abspath(__file__)).read()
DAY_HRS = {0: 8.5, 1: 8.5, 2: 8.5, 3: 8.5, 4: 6.0}
def clean(v):
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Period): return str(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [{"label": "report3.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]
    json.dump({"rows": rows, "meta": meta}, open(f"{OUT}/z_{name}.json", "w"), allow_nan=False); print("wrote z_" + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]

d = pd.read_parquet(SRC); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); END = d.date.max()
LINES = ["ASF", "ASC", "GSL", "AIRSET SPURLINE", "DIE"]; FOUNDRY = LINES + ["FETT", "DESP"]
NM = {"ASF": "Moulding ASF", "ASC": "Moulding ASC", "GSL": "Moulding GSL", "AIRSET SPURLINE": "Moulding Airset", "DIE": "Moulding DIE", "FETT": "Fettling", "DESP": "Casting despatch", "MDESP": "Machined despatch"}
ann = 365 / END.dayofyear
# ---- attendance model (same rules as report.py), kept at operator level
lb = d[d.date.dt.dayofweek < 5].copy()
timed = lb[lb.hrs_clean > 0].groupby(["date", "name"]).hrs_clean.sum().clip(upper=12)
ua = lb[lb.work_centre.isin(NM)]
cnt = ua.groupby(["date", "name", "work_centre"]).size().rename("n").reset_index()
cnt = cnt.join(timed.rename("timed"), on=["date", "name"]).join(lb[lb.hrs_clean.fillna(0) == 0].groupby(["date", "name"]).size().rename("un"), on=["date", "name"])
cnt["hrs"] = (cnt.date.dt.dayofweek.map(DAY_HRS) - cnt.timed.fillna(0)).clip(lower=0) * cnt.n / cnt.un.clip(lower=1)
ah = cnt.groupby(["work_centre", "date"]).hrs.sum().rename("hrs").reset_index()
pq = ua.groupby(["work_centre", "date", "part_no"]).qty.sum().reset_index()
pq = pq.join(pq.groupby(["work_centre", "date"]).qty.sum().rename("dq"), on=["work_centre", "date"]).merge(ah, on=["work_centre", "date"], how="left")
pq["alloc"] = pq.hrs * pq.qty / pq.dq.replace(0, np.nan)
std = pq.groupby(["work_centre", "part_no"]).agg(u=("qty", "sum"), h=("alloc", "sum")); std["t"] = std.h / std.u.replace(0, np.nan)
pq = pq.join(std.t, on=["work_centre", "part_no"]); pq["earned"] = pq.qty * pq.t
day = pq.groupby(["work_centre", "date"]).agg(earned=("earned", "sum"), units=("qty", "sum"), hrs=("hrs", "first"), n_parts=("part_no", "nunique")).reset_index(); day["yr"] = day.date.dt.year
wd = pd.Series(sorted(d[d.date.dt.dayofweek < 5].date.unique())); wdy = wd.groupby(wd.dt.year).size()
# ---- orders
q = d[d.op.notna()].groupby(["wo_base", "op"]).qty.sum().unstack()
def col(lo, hi):
    c = [x for x in q.columns if lo <= x <= hi]; return q[c].sum(axis=1, min_count=1)
m_, f_, dsp = col(10, 19), col(20, 59), col(80, 99)
fs = d[d.op.between(10, 19)].groupby("wo_base").date.min(); ff = d[d.op.between(20, 59)].groupby("wo_base").date.min(); fd_ = d[d.op.between(80, 99)].groupby("wo_base").date.min()
o = pd.DataFrame({"mould": fs, "fettle": ff, "desp": fd_}).dropna(subset=["mould"])
o["qty"] = m_.reindex(o.index); o = o[o.qty > 0]
o["line"] = d[d.op.between(10, 19)].sort_values("date").groupby("wo_base").work_centre.first().reindex(o.index)
o["part"] = d.groupby("wo_base").part_no.first().reindex(o.index)
o["wait_mf"] = (o.fettle - o.mould).dt.days; o["lead"] = (o.desp - o.mould).dt.days
o["good"] = dsp.reindex(o.index).clip(upper=o.qty); o["yield"] = o.good / o.qty  # orders with no despatch booking carry no yield, as in report.py
o["mature"] = (o.mould < END - pd.Timedelta(days=45)) & o.good.notna()
o["size"] = [std.t.get((l, p), np.nan) for l, p in zip(o.line, o.part)]
o = o.sort_values("mould"); o["rank"] = o.groupby("part").cumcount() + 1; o["yr"] = o.mould.dt.year
mo = d[d.op.between(10, 19) & d.work_centre.isin(LINES)]
size_cls = lambda t: "Small" if t < 0.1 else "Medium" if t <= 0.4 else "Large"

# ===== 1. reprice list
cy = {int(r["year"]): r for r in J("x_commercial_year")}; rate = {y: r["loaded_min_rate"] for y, r in cy.items()}
fh = pq[pq.work_centre.isin(FOUNDRY)].groupby("part_no").agg(h=("alloc", "sum"))
pu = mo.groupby("part_no").agg(units=("qty", "sum"), first=("date", "min"), last=("date", "max"))
u12 = mo[mo.date > END - pd.Timedelta(days=365)].groupby("part_no").qty.sum()
rp = pu.join(fh).join(u12.rename("units_12m")).fillna({"units_12m": 0}); rp = rp[(rp.units >= 20) & (rp.units_12m > 0)]
rp["hrs_per_casting"] = rp.h / rp.units; rp["first_year"] = rp["first"].dt.year
rp["cost_first"] = rp.hrs_per_casting * rp.first_year.map(rate); rp["cost_now"] = rp.hrs_per_casting * rate[2026]; rp["cost_system_rate"] = rp.hrs_per_casting * 11.0
rp["uplift_pct"] = 100 * (rp.cost_now / rp.cost_first - 1); rp["annual_gap_gbp"] = (rp.cost_now - rp.cost_first) * rp.units_12m; rp["annual_gap_vs_system_gbp"] = (rp.cost_now - rp.cost_system_rate) * rp.units_12m
rp["line"] = mo.groupby("part_no").work_centre.agg(lambda s: s.mode().iat[0]).reindex(rp.index)
rpo = rp.sort_values("annual_gap_gbp", ascending=False).reset_index()[["part_no", "line", "first_year", "units", "units_12m", "hrs_per_casting", "cost_first", "cost_system_rate", "cost_now", "uplift_pct", "annual_gap_gbp", "annual_gap_vs_system_gbp"]]
save("reprice", rpo.head(60), "Foundry labour per casting (moulding, fettling, despatch attendance hours) costed at the statutory minimum loaded rate in the part's first year and in 2026. Annual gap = cost increase x castings moulded in the last 12 months. Parts with 20+ castings and activity in the last 12 months. Labour only: no metal, energy or overhead.")
save("reprice_all", rpo, "All active parts: foundry labour hours per casting and cost at first-year and 2026 minimum loaded rates.")
save("reprice_summary", [{"parts": len(rp), "castings_12m": rp.units_12m.sum(), "labour_now_gbp": (rp.cost_now * rp.units_12m).sum(), "gap_vs_first_year_gbp": rp.annual_gap_gbp.sum(), "gap_vs_system_rate_gbp": rp.annual_gap_vs_system_gbp.sum(), "parts_first_made_before_2024": int((rp.first_year < 2024).sum()), "gap_from_pre_2024_parts_gbp": rp[rp.first_year < 2024].annual_gap_gbp.sum(), "top20_share_pct": 100 * rpo.head(20).annual_gap_gbp.sum() / rp.annual_gap_gbp.sum()}])

# ===== 2. part churn
py = mo.groupby(["part_no", mo.date.dt.year]).qty.sum().unstack(fill_value=0); py[2026] = py[2026] * ann
t_m = std.t.reset_index(); t_m = t_m[t_m.work_centre.isin(LINES)].groupby("part_no").t.mean()
ch = []
for y in range(2022, 2027):
    a, b = py[y - 1], py[y]
    new, gone, cont = (a == 0) & (b > 0), (a > 0) & (b == 0), (a > 0) & (b > 0)
    ch.append({"year": str(y), "parts_made": int((b > 0).sum()), "new_parts": int(new.sum()), "units_from_new": b[new].sum(), "dropped_parts": int(gone.sum()), "units_lost_from_dropped": a[gone].sum(), "continuing_parts": int(cont.sum()), "continuing_change_units": (b[cont] - a[cont]).sum(), "net_change_units": b.sum() - a.sum()})
save("churn_year", ch, "Parts moulded each year against the year before. New = not made the year before; dropped = made the year before, not this year. 2026 annualised.")
dl = (py[2026] - py[2023]).sort_values()
def crow(p): return {"part_no": p, "line": rp.line.get(p) or mo[mo.part_no == p].work_centre.mode().iat[0], "size": size_cls(t_m.get(p, np.nan)) if t_m.get(p, np.nan) == t_m.get(p, np.nan) else None, "units_2023": py.loc[p, 2023], "units_2026_annualised": py.loc[p, 2026], "change": dl[p], "last_made": mo[mo.part_no == p].date.max()}
save("churn_losers", [crow(p) for p in dl.head(15).index], "Fifteen parts with the largest fall in castings, 2023 to 2026 (annualised).")
save("churn_gainers", [crow(p) for p in dl.tail(15).index[::-1]], "Fifteen parts with the largest rise in castings, 2023 to 2026 (annualised).")
save("churn_summary", [{"net_change": dl.sum(), "fall_from_top15": dl.head(15).sum(), "top15_share_of_gross_fall_pct": 100 * dl.head(15).sum() / dl[dl < 0].sum(), "gross_fall": dl[dl < 0].sum(), "gross_rise": dl[dl > 0].sum(), "parts_2023": int((py[2023] > 0).sum()), "parts_2026": int((py[2026] > 0).sum()), "parts_2023_not_2026": int(((py[2023] > 0) & (py[2026] == 0)).sum()), "units_2023_from_parts_gone": py[2023][(py[2026] == 0)].sum()}])

# ===== 3. new vs repeat
bk = lambda r: "1st order" if r == 1 else "2nd" if r == 2 else "3rd to 5th" if r <= 5 else "6th to 20th" if r <= 20 else "21st on"
o["bucket"] = o["rank"].map(bk)
pqo = d[d.work_centre.isin(LINES) & (d.date.dt.dayofweek < 5)].groupby(["work_centre", "date", "part_no", "wo_base"]).qty.sum().reset_index()
pqo = pqo.merge(pq[["work_centre", "date", "part_no", "qty", "alloc", "t"]].rename(columns={"qty": "pqty"}), on=["work_centre", "date", "part_no"])
pqo["alloc_o"] = pqo.alloc * pqo.qty / pqo.pqty.replace(0, np.nan); pqo["earned_o"] = pqo.qty * pqo.t
oe = pqo.groupby("wo_base").agg(alloc=("alloc_o", "sum"), earned=("earned_o", "sum")); o = o.join(oe)
multi = o[o.part.map(o.groupby("part").size()) >= 6]
nr = []
for b in ["1st order", "2nd", "3rd to 5th", "6th to 20th", "21st on"]:
    for scope, g in (("all parts", o[o.bucket == b]), ("parts with 6+ orders", multi[multi.bucket == b])):
        gm = g[g.mature & (g.mould < "2026-08-01")]
        nr.append({"order": b, "scope": scope, "orders": len(g), "yield_pct": 100 * gm.good.sum() / gm.qty.sum(), "orders_short_pct": 100 * (gm["yield"] < 1).mean(), "mould_to_fettle_median": g.wait_mf[(g.wait_mf >= 0) & (g.wait_mf <= 365)].median(), "lead_median": g.lead[(g.lead >= 0) & (g.lead <= 365)].median(), "moulding_hrs_vs_part_norm_pct": 100 * g.alloc.sum() / g.earned.sum(), "median_batch": g.qty.median()})
save("new_vs_repeat", nr, "Casting orders grouped by how many times the part had been ordered. Moulding hours vs part norm: attendance hours used against the part's own all-years average (100 = norm). Yield uses orders started before August 2026.")

# ===== 4 and 5. prediction models
o["month"] = o.mould.dt.month; o["dow"] = o.mould.dt.dayofweek
mv, fv = o.mould.values, o.fettle.fillna(pd.Timestamp("2100-01-01")).values; qv = o.qty.values
openq = np.zeros(len(o)); opens = np.zeros(len(o))
for i in range(len(o)):
    m = (mv < mv[i]) & (fv > mv[i]) & (mv > mv[i] - np.timedelta64(90, "D")); openq[i] = qv[m].sum(); opens[i] = m.sum()
o["queue_units"] = openq; o["queue_orders"] = opens
ok_lead = o.lead.where((o.lead >= 0) & (o.lead <= 365)); ok_y = o["yield"].where(o.mature & (o.mould < "2026-07-01"))
o["prev_lead"] = ok_lead.groupby(o.part).transform(lambda s: s.shift().expanding().median()); o["prev_yield"] = ok_y.groupby(o.part).transform(lambda s: s.shift().expanding().mean())
o["prev_short"] = (ok_y < 0.97).astype(float).where(ok_y.notna()).groupby(o.part).transform(lambda s: s.shift().expanding().mean())
o["line_c"] = o.line.astype("category").cat.codes; o["t_days"] = (o.mould - pd.Timestamp("2021-01-01")).dt.days
o["line_recent_lead"] = np.nan
for l, g in o.groupby("line"):
    s = pd.Series(ok_lead.reindex(g.index).values, index=g.mould.values).shift().rolling(20, min_periods=5).median(); o.loc[g.index, "line_recent_lead"] = s.values
FL = ["line_c", "qty", "size", "rank", "month", "dow", "queue_units", "queue_orders", "prev_lead", "prev_yield", "prev_short", "line_recent_lead"]
LAB = {"line_c": "Moulding line", "qty": "Batch size", "size": "Casting size", "rank": "Times part ordered before", "month": "Month", "dow": "Weekday started", "queue_units": "Castings waiting to fettle at start", "queue_orders": "Orders waiting to fettle at start", "prev_lead": "Part's past lead time", "prev_yield": "Part's past yield", "prev_short": "Part's past short-order rate", "line_recent_lead": "Line's last 20 lead times"}
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=12, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, deterministic=True, n_jobs=1, verbose=-1, force_row_wise=True)
L = o[ok_lead.notna()].copy(); L["y"] = np.log1p(ok_lead[L.index]); CUT = pd.Timestamp("2025-01-01")
tr, te = L[L.mould < CUT], L[(L.mould >= CUT) & (L.mould < "2026-08-01")]
pr = {}
for a_ in (0.1, 0.5, 0.9): pr[a_] = np.expm1(lgb.LGBMRegressor(objective="quantile", alpha=a_, **P).fit(tr[FL], tr.y).predict(te[FL]))
act = np.expm1(te.y.values); base_g = np.full(len(te), np.expm1(tr.y).median()); base_p = te.prev_lead.fillna(np.expm1(tr.y).median()).values
mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[FL], tr.y); imp = pd.Series(mdl.booster_.feature_importance("gain"), index=FL); imp = 100 * imp / imp.sum()
save("lead_model", [{"train_orders": len(tr), "test_orders": len(te), "test_period": "Jan 2025 to Jul 2026", "mae_model_days": np.abs(act - pr[0.5]).mean(), "mae_part_history_days": np.abs(act - base_p).mean(), "mae_overall_median_days": np.abs(act - base_g).mean(), "median_abs_error_model_days": np.median(np.abs(act - pr[0.5])), "band_80_coverage_pct": 100 * ((act >= pr[0.1]) & (act <= pr[0.9])).mean(), "median_band_width_days": np.median(pr[0.9] - pr[0.1]), "late_beyond_p90_pct": 100 * (act > pr[0.9]).mean()}], "LightGBM quantile regression (10th, 50th, 90th percentile) of casting lead time in days from first moulding to first despatch booking, using only what is known at launch. Trained on 2021 to 2024 starts, tested on later starts.")
save("lead_importance", [{"factor": LAB[k], "share_pct": v} for k, v in imp.sort_values(ascending=False).items()], "Share of model gain by input, lead-time model.")
te = te.assign(p50=pr[0.5], p10=pr[0.1], p90=pr[0.9], act=act); te["band"] = pd.qcut(te.p50, 5, labels=["Fastest fifth", "2", "3", "4", "Slowest fifth"])
save("lead_calibration", [{"predicted_group": str(b), "orders": len(g), "predicted_median_days": g.p50.median(), "actual_median_days": g.act.median(), "predicted_p90_days": g.p90.median(), "actual_p90_days": g.act.quantile(.9)} for b, g in te.groupby("band")], "Test orders grouped by predicted lead time: predicted against actual.")
op_ = o[o.desp.isna() & (o.mould > END - pd.Timedelta(days=120))].copy()
if len(op_):
    full = {a_: lgb.LGBMRegressor(objective="quantile", alpha=a_, **P).fit(L[FL], L.y) for a_ in (0.5, 0.9)}
    op_["p50"] = np.expm1(full[0.5].predict(op_[FL])); op_["p90"] = np.expm1(full[0.9].predict(op_[FL])); op_["age"] = (END - op_.mould).dt.days
    op_["status"] = np.where(op_.age > op_.p90, "Past 90th percentile", np.where(op_.age > op_.p50, "Past median", "Within median"))
    save("lead_open_orders", [{"status": s, "orders": len(g), "castings": g.qty.sum(), "median_age_days": g.age.median()} for s, g in op_.groupby("status")], "Casting orders moulded in the last 120 days with no despatch booking, against the model's expected lead time. Affected by the despatch booking gap since August 2026.")
    save("lead_open_list", op_[op_.status == "Past 90th percentile"].sort_values("age", ascending=False).head(40).reset_index()[["wo_base", "part", "line", "qty", "mould", "age", "p50", "p90"]].rename(columns={"wo_base": "works_order", "mould": "moulded", "age": "age_days", "p50": "expected_days", "p90": "late_after_days"}), "Open casting orders older than the model's 90th percentile lead time.")
S = o[ok_y.notna()].copy(); S["y"] = (ok_y[S.index] < 0.97).astype(int); FS = [f for f in FL if f not in ("queue_units", "queue_orders", "line_recent_lead")]
tr, te = S[S.mould < CUT], S[S.mould >= CUT]
PC = {k: v for k, v in P.items()}; clf = lgb.LGBMClassifier(**PC).fit(tr[FS], tr.y); ps = clf.predict_proba(te[FS])[:, 1]
te = te.assign(p=ps); te["grp"] = pd.qcut(te.p.rank(method="first"), 5, labels=["Lowest risk fifth", "2", "3", "4", "Highest risk fifth"])
imp = pd.Series(clf.booster_.feature_importance("gain"), index=FS); imp = 100 * imp / imp.sum()
lost_te = (te.qty - te.good)
save("scrap_model", [{"train_orders": len(tr), "test_orders": len(te), "test_period": "Jan 2025 to Jun 2026", "short_rate_train_pct": 100 * tr.y.mean(), "short_rate_test_pct": 100 * te.y.mean(), "auc": roc_auc_score(te.y, ps), "auc_part_history_only": roc_auc_score(te.y, te.prev_short.fillna(tr.y.mean())), "top_fifth_short_rate_pct": 100 * te[te.grp == "Highest risk fifth"].y.mean(), "bottom_fifth_short_rate_pct": 100 * te[te.grp == "Lowest risk fifth"].y.mean(), "top_fifth_share_of_lost_castings_pct": 100 * lost_te[te.grp == "Highest risk fifth"].sum() / lost_te.sum()}], "LightGBM classifier: probability that a casting order despatches under 97% of the quantity moulded, from what is known before moulding. Trained on 2021 to 2024, tested on Jan 2025 to Jun 2026. AUC 0.5 = no skill, 1.0 = perfect.")
save("scrap_groups", [{"risk_group": str(b), "orders": len(g), "predicted_risk_pct": 100 * g.p.mean(), "actual_short_pct": 100 * g.y.mean(), "castings_moulded": g.qty.sum(), "castings_lost": (g.qty - g.good).sum()} for b, g in te.groupby("grp")], "Test orders in five equal groups by predicted risk.")
save("scrap_importance", [{"factor": LAB[k], "share_pct": v} for k, v in imp.sort_values(ascending=False).items()], "Share of model gain by input, short-order model.")
clf_all = lgb.LGBMClassifier(**PC).fit(S[FS], S.y); act_p = o[o.mould > END - pd.Timedelta(days=365)].groupby("part").tail(1).copy()
act_p["rank"] = act_p["rank"] + 1; nxt = act_p.copy()
for c_ in ("prev_yield", "prev_short"):
    nxt[c_] = {"prev_yield": ok_y, "prev_short": (ok_y < 0.97).astype(float).where(ok_y.notna())}[c_].groupby(o.part).mean().reindex(nxt.part).values
nxt["prev_lead"] = ok_lead.groupby(o.part).median().reindex(nxt.part).values; nxt["month"] = 12; nxt["dow"] = 0
nxt["risk"] = 100 * clf_all.predict_proba(nxt[FS])[:, 1]; nxt["orders"] = nxt.part.map(o.groupby("part").size()).values; nxt["past_yield_pct"] = 100 * nxt.prev_yield
save("scrap_watchlist", nxt.sort_values("risk", ascending=False).head(25)[["part", "line", "orders", "qty", "past_yield_pct", "risk"]].rename(columns={"part": "part_no", "qty": "last_batch", "risk": "risk_next_order_pct"}), "Parts moulded in the last 12 months with the highest predicted chance that the next order comes up short (model refitted on all orders).")

# ===== 6. control charts
cc, sig = [], []
for wc in LINES + ["FETT"]:
    g = day[day.work_centre == wc].set_index("date")
    w = g.earned.resample("W").sum(); w = w[(w.index >= "2021-01-10") & (w.index < END - pd.Timedelta(days=6))]; w = w[w > 0]
    base = w[w.index < "2025-01-01"]; cl = base.mean(); mr = base.diff().abs().mean(); ucl, lcl = cl + 2.66 * mr, max(cl - 2.66 * mr, 0)
    side = np.sign(w - cl); run = side.groupby((side != side.shift()).cumsum()).cumcount() + 1
    flag = np.where(w > ucl, "above limit", np.where(w < lcl, "below limit", np.where((run >= 8) & (side < 0), "run below average", np.where((run >= 8) & (side > 0), "run above average", ""))))
    for t, v, f in zip(w.index, w.values, flag):
        if t >= pd.Timestamp("2024-01-01"): cc.append({"area": NM[wc], "week": t, "std_hrs": v, "centre": cl, "ucl": ucl, "lcl": lcl, "signal": f})
    w26 = pd.Series(flag, index=w.index)[w.index >= "2026-01-01"]; last = w[w.index >= "2026-01-01"]
    sig.append({"area": NM[wc], "baseline_mean_std_hrs_wk": cl, "lower_limit": lcl, "upper_limit": ucl, "mean_2026": last.mean(), "weeks_2026": len(last), "weeks_below_limit_2026": int((w26 == "below limit").sum()), "weeks_above_limit_2026": int((w26 == "above limit").sum()), "weeks_in_low_run_2026": int((w26 == "run below average").sum()), "weeks_in_high_run_2026": int((w26 == "run above average").sum()), "latest_signal": next((f for f in w26.values[::-1] if f), "none"), "shifted": bool(((w26 != "").mean() if len(w26) else 0) > 0.25)})
save("control_output", cc, "Individuals (XmR) control chart of weekly standard hours earned per area. Centre line and limits (mean plus or minus 2.66 x average moving range) from 2021 to 2024. Signals: a week outside the limits, or 8 weeks running on one side of the centre line.")
save("control_output_summary", sig, "Control chart signals in 2026 by area.")
oy = o[o.mature & (o.mould < "2026-08-01") & o.line.isin(LINES)].copy(); oy["q"] = oy.mould.dt.to_period("Q"); py_ = []
for wc in LINES + ["All lines"]:
    g = oy if wc == "All lines" else oy[oy.line == wc]; b = g[g.mould < "2025-01-01"]; pbar = 1 - b.good.sum() / b.qty.sum()
    for qq, x in g.groupby("q"):
        n = x.qty.sum(); p_ = 1 - x.good.sum() / n; se = math.sqrt(pbar * (1 - pbar) / n) * math.sqrt(max(1.0, ((b.qty - b.good) / b.qty).var() * b.qty.mean() / (pbar * (1 - pbar))))
        py_.append({"line": NM.get(wc, wc), "quarter": str(qq), "moulded": n, "loss_pct": 100 * p_, "centre_pct": 100 * pbar, "ucl_pct": 100 * (pbar + 3 * se), "signal": "above limit" if p_ > pbar + 3 * se else ""})
save("control_yield", py_, "Laney-adjusted p-chart of castings lost (moulded less despatched) per quarter by line, orders started before August 2026. Centre line from 2021 to 2024 orders; limits widened for between-order variation.")

# ===== 7. capacity scenarios
FTE_H = 39 * 46
cap = []
for wc in LINES:
    g = day[day.work_centre == wc]; g26 = g[g.yr == 2026]; g23 = g[g.yr == 2023]
    hps = g26.hrs.sum() / g26.earned.sum() if g26.earned.sum() else np.nan; p90 = g[g.yr <= 2025].earned.quantile(.9); mean_day = g26.earned.mean()
    s26, s23 = g26.earned.sum() * ann, g23.earned.sum()
    cap.append({"line": NM[wc], "std_hrs_2026_annualised": s26, "std_hrs_2023": s23, "labour_hrs_per_std_hr_2026": hps, "labour_hrs_2026_annualised": g26.hrs.sum() * ann, "labour_hrs_for_2023_mix": s23 * hps, "extra_fte_for_2023_mix": (s23 * hps - g26.hrs.sum() * ann) / FTE_H,
                "producing_days_2026_annualised": len(g26) * ann, "days_for_2023_mix_at_2026_rate": s23 / mean_day if mean_day else None, "days_for_2023_mix_at_best_rate": s23 / p90, "working_days": int(wdy.get(2025, 252)), "fits_at_2026_rate": bool(s23 / mean_day <= wdy.get(2025, 252)) if mean_day else None})
save("capacity_2023_mix", cap, "Labour and days each moulding line would need to make its 2023 output again, at 2026 labour per standard hour and 2026 average daily output. FTE = 39 hours x 46 weeks.")
dd_ = day[(day.yr == 2026) & day.work_centre.isin(["DIE", "GSL"])]; both = dd_.groupby("date").work_centre.nunique()
ppl = d[(d.yr == 2026) & d.work_centre.isin(["DIE", "GSL"])].groupby("work_centre").name.agg(lambda s: set(s))
save("capacity_die_gsl", [{"working_days_2026_to_date": int(wdy.get(2026, 0)), "die_producing_days": int((dd_.work_centre == "DIE").sum()), "gsl_producing_days": int((dd_.work_centre == "GSL").sum()), "days_both_ran": int((both == 2).sum()), "days_either_ran": int(len(both)), "days_neither_ran": int(wdy.get(2026, 0) - len(both)),
                           "die_labour_hrs": dd_[dd_.work_centre == "DIE"].hrs.sum(), "gsl_labour_hrs": dd_[dd_.work_centre == "GSL"].hrs.sum(), "combined_fte": dd_.hrs.sum() * ann / FTE_H, "people_booking_die": len(ppl.get("DIE", set())), "people_booking_gsl": len(ppl.get("GSL", set())), "people_booking_both": len(ppl.get("DIE", set()) & ppl.get("GSL", set())),
                           "combined_std_hrs": dd_.earned.sum(), "days_needed_at_gsl_best_rate": dd_.earned.sum() / day[(day.work_centre == "GSL") & (day.yr <= 2025)].earned.quantile(.9)}], "DIE and GSL in 2026: days each produced and whether one crew alternating between them could cover both. Arithmetic on bookings only; says nothing about process or tooling constraints.")

# ===== 8. calendar effects
dn = ["Mon", "Tue", "Wed", "Thu", "Fri"]; cal = []
for grp, wcs in (("Moulding (all lines)", LINES), ("Fettling", ["FETT"])):
    g = day[day.work_centre.isin(wcs)].groupby("date").agg(earned=("earned", "sum"), hrs=("hrs", "sum")).reset_index(); g["dow"] = g.date.dt.dayofweek; g["per_h"] = g.earned / g.hrs.replace(0, np.nan)
    kw = stats.kruskal(*[g[g.dow == i].per_h.dropna() for i in range(5)])
    for i in range(5):
        x = g[g.dow == i]; cal.append({"area": grp, "weekday": dn[i], "days": len(x), "std_hrs_per_day": x.earned.mean(), "labour_hrs_per_day": x.hrs.mean(), "std_hrs_per_labour_hr": x.earned.sum() / x.hrs.sum(), "index_vs_week": 100 * (x.earned.sum() / x.hrs.sum()) / (g.earned.sum() / g.hrs.sum()), "kruskal_p": kw.pvalue})
save("calendar_weekday", cal, "Standard hours earned per day and per attendance hour by weekday (Friday is a 6-hour day). Kruskal-Wallis test across weekdays.")
om = o[o.mature & (o.mould < "2026-08-01")]; 
save("calendar_yield_weekday", [{"weekday_moulded": dn[i], "orders": len(g), "moulded": g.qty.sum(), "yield_pct": 100 * g.good.sum() / g.qty.sum(), "orders_short_pct": 100 * (g["yield"] < 0.97).mean()} for i, g in om[om.dow < 5].groupby("dow")], "Casting yield by the weekday the order was first moulded.")
ms = day[day.work_centre.isin(LINES) & (day.yr <= 2025)].groupby([day.yr, day.date.dt.month]).earned.sum().unstack(0); mi = ms.div(ms.mean()).mean(axis=1) * 100
dm = day[day.work_centre.isin(LINES) & (day.yr <= 2025)].groupby([day.yr, day.date.dt.month]).date.nunique().unstack(0).mean(axis=1)
save("calendar_month", [{"month": pd.Timestamp(2025, m, 1).strftime("%b"), "output_index": v, "producing_days_avg": dm[m]} for m, v in mi.items()], "Moulding standard hours by month as an index of the average month, 2021 to 2025.")
g = day[day.work_centre.isin(LINES)].groupby("date").earned.sum(); dom = pd.Series(g.index.day, index=g.index)
save("calendar_month_part", [{"part_of_month": n, "days": int(m.sum()), "std_hrs_per_day": g[m.values].mean(), "index": 100 * g[m.values].mean() / g.mean()} for n, m in (("Days 1 to 10", dom <= 10), ("Days 11 to 20", (dom > 10) & (dom <= 20)), ("Days 21 to end", dom > 20))], "Moulding standard hours per producing day by part of the month.")

# ===== 9. variety cost per day
import statsmodels.formula.api as smf
vr = []
for wc in LINES:
    g = day[(day.work_centre == wc) & (day.hrs > 2) & (day.earned > 0)].copy(); g["eff"] = g.earned / g.hrs; g = g[(g.eff > g.eff.quantile(.02)) & (g.eff < g.eff.quantile(.98))]; g["le"] = np.log(g.eff); g["lh"] = np.log(g.hrs)
    m = smf.ols("le ~ n_parts + lh + C(yr)", g).fit(cov_type="HC1")
    vr.append({"line": NM[wc], "days": len(g), "median_parts_per_day": g.n_parts.median(), "p90_parts_per_day": g.n_parts.quantile(.9), "output_change_per_extra_part_pct": 100 * (math.exp(m.params["n_parts"]) - 1), "ci_low": 100 * (math.exp(m.conf_int().loc["n_parts", 0]) - 1), "ci_high": 100 * (math.exp(m.conf_int().loc["n_parts", 1]) - 1), "p_value": m.pvalues["n_parts"],
               "std_hrs_per_labour_hr_1_2_parts": g[g.n_parts <= 2].earned.sum() / g[g.n_parts <= 2].hrs.sum(), "std_hrs_per_labour_hr_5plus_parts": g[g.n_parts >= 5].earned.sum() / g[g.n_parts >= 5].hrs.sum() if (g.n_parts >= 5).sum() > 10 else None})
save("variety_cost", vr, "Regression of log(standard hours earned per attendance hour) on the number of different parts a line ran that day, controlling for hours worked and year (robust standard errors). Standard times are each part's own average, so a part normally run on busy mixed days already carries some changeover time; the effect shown is what remains.")

# ===== 10. notes and lost time
lt = d[d.type.isin(["Lost", "Rework"])].copy(); _tx = lt.lost_time.astype(str).where(lt.lost_time.notna() & ~lt.lost_time.astype(str).str.fullmatch(r"[\d.]+")); _mp = lt.assign(tx=_tx).dropna(subset=["tx"]).groupby("lost_code").tx.agg(lambda s_: s_.mode().iat[0]) if _tx.notna().any() else {}
lt["code"] = _tx.fillna(lt.lost_code.map(_mp)).fillna(lt.lost_code.astype(str)).fillna("Not coded").astype(str).str.strip()
lc = lt.groupby("code").agg(entries=("qty", "size"), hrs=("hrs_clean", "sum"), first=("date", "min"), last=("date", "max")).sort_values("hrs", ascending=False).head(15).reset_index()
ly = lt.groupby(["code", "yr"]).hrs_clean.sum().unstack(fill_value=0)
for y in range(2021, 2027): lc[f"hrs_{y}"] = lc.code.map(ly[y]) if y in ly else 0.0
save("lost_codes", lc, "Lost-time and rework bookings by reason code: entries, clean hours, and hours by year.")
names = set(w for n in d.name.dropna().unique() for w in re.findall(r"[a-z]+", str(n).lower()))
STOP = set("the and for with from this that was were had has have not are but all out off per one two ref see due new job jobs part parts into then when only also done made make used using after before each been over".split()) | names
def toks(s): return [w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in STOP]
tk = []
for grp, sel in (("Lost time and rework", d.type.isin(["Lost", "Rework"])), ("Moulding", d.work_centre.isin(LINES)), ("Fettling and despatch", d.work_centre.isin(["FETT", "DESP", "MDESP"])), ("Machining", d.op >= 100)):
    x = d[sel & d.notes.notna()]; c_ = pd.Series([w for s in x.notes for w in set(toks(s))]).value_counts().head(15)
    for w, n in c_.items(): tk.append({"group": grp, "term": w, "entries": int(n), "share_of_noted_entries_pct": 100 * n / max(len(x), 1), "hrs": x[x.notes.str.lower().str.contains(r"\b" + w + r"\b", regex=True)].hrs_clean.sum()})
save("note_terms", tk, "Most common words in free-text notes by group (words of 4+ letters; operator names and initials removed). Share is of entries that carry a note.")
nt = mo[mo.notes.notna()].copy(); nt["ref"] = nt.notes.str.strip().str.upper().str.extract(r"^(F1|F2|F|B|A|K|M|C)")[0]
nc = nt.dropna(subset=["ref"]).groupby(["ref", "work_centre"]).size().unstack(fill_value=0); nyr = nt.dropna(subset=["ref"]).groupby(["ref", nt.date.dt.year]).size().unstack(fill_value=0)
save("note_codes", [{"prefix": r, "entries": int(nc.loc[r].sum()), "main_line": NM[nc.loc[r].idxmax()], "main_line_share_pct": 100 * nc.loc[r].max() / nc.loc[r].sum(), "first_year": int(nyr.loc[r][nyr.loc[r] > 0].index.min()), "last_year": int(nyr.loc[r][nyr.loc[r] > 0].index.max())} for r in nc.sum(axis=1).sort_values(ascending=False).index], "Moulding note prefixes (meaning not confirmed) by line and years in use.")
save("note_coverage", [{"year": str(y), "entries": len(g), "with_note_pct": 100 * g.notes.notna().mean(), "moulding_with_prefix_pct": 100 * g[g.work_centre.isin(LINES)].notes.fillna("").str.strip().str.upper().str.match(r"^(F1|F2|F|B|A|K|M|C)").mean()} for y, g in d.groupby("yr")], "Share of entries with a note, by year.")

# ===== 11. booking discipline
d["grp"] = np.where(d.work_centre.isin(LINES), "Moulding", np.where(d.work_centre == "FETT", "Fettling", np.where(d.work_centre.isin(["DESP", "MDESP"]), "Despatch", np.where(d.op >= 100, "Machining", "Other"))))
bd = []
for (gname, y), g in d.groupby(["grp", "yr"]):
    att = g.groupby("date").name.nunique().sum()
    bd.append({"area": gname, "year": str(y), "entries": len(g), "no_time_pct": 100 * (g.hrs_clean.fillna(0) == 0).mean(), "hours_disagree_with_clock_pct": 100 * g.flag_hrs_mismatch.fillna(False).astype(bool).mean(), "zero_quantity_pct": 100 * (g.qty.fillna(0) == 0).mean(), "no_works_order_pct": 100 * g.wo_base.isna().mean(), "with_note_pct": 100 * g.notes.notna().mean(), "weekend_pct": 100 * (g.date.dt.dayofweek >= 5).mean(), "entries_per_person_day": len(g) / max(att, 1)})
save("booking_discipline", bd, "Booking quality by area and year. The export holds the work date only, not when the booking was keyed, so late booking cannot be measured.")
wdays = pd.DataFrame({"date": wd}); wdays["ym"] = wdays.date.dt.to_period("M"); gap = []
for gname, wcs in (("Moulding", LINES), ("Fettling", ["FETT"]), ("Casting despatch", ["DESP"]), ("Machined despatch", ["MDESP"])):
    act_d = set(d[d.work_centre.isin(wcs)].date.unique()); wdays[gname] = wdays.date.isin(act_d)
gm_ = wdays[wdays.date >= "2025-01-01"].groupby("ym")[["Moulding", "Fettling", "Casting despatch", "Machined despatch"]].agg(lambda s: int((~s).sum()))
save("booking_gaps", [{"month": str(i), **{c: int(r[c]) for c in gm_.columns}} for i, r in gm_.iterrows()], "Working days with no booking at all in each area, by month since January 2025.")

# ===== 12. operators (codes, no names)
fo = cnt[cnt.work_centre.isin(LINES + ["FETT"])].copy(); un = ua[ua.work_centre.isin(LINES + ["FETT"])].groupby(["date", "name", "work_centre", "part_no"]).qty.sum().reset_index().join(std.t, on=["work_centre", "part_no"])
un["earned"] = un.qty * un.t; ed = un.groupby(["date", "name", "work_centre"]).agg(earned=("earned", "sum"), units=("qty", "sum")).reset_index(); fo = fo.merge(ed, on=["date", "name", "work_centre"], how="left")
# crews: one person often books for a team, so score the crew-day and attribute by presence
first = d.groupby("name").date.min(); fo["tenure_m"] = ((fo.date - fo.name.map(first)).dt.days / 30.4)
tb = pd.cut(fo.tenure_m, [-1, 3, 6, 12, 24, 48, 999], labels=["0 to 3 months", "3 to 6", "6 to 12", "1 to 2 years", "2 to 4 years", "4 years on"])
started = fo.name.map(first) > "2021-03-01"
save("operator_tenure", [{"time_since_first_booking": str(b), "operators": g.name.nunique(), "person_days": len(g), "std_hrs_per_labour_hr": g.earned.sum() / g.hrs.sum(), "units_per_day": g.units.sum() / len(g)} for b, g in fo[started & (fo.hrs > 0)].groupby(tb) if len(g) >= 20], "Standard hours booked per attendance hour by time since the operator's first booking, for operators who first appear after March 2021 (moulding and fettling). One person often books a crew's output, so this measures who books, not individual pace.")
codes = {n: f"Operator {i + 1:02d}" for i, n in enumerate(fo.groupby("name").hrs.sum().sort_values(ascending=False).index)}
opr = []
for (n, wc), g in fo[fo.hrs > 0].groupby(["name", "work_centre"]):
    if len(g) < 60: continue
    g26 = g[g.date.dt.year == 2026]
    opr.append({"operator": codes[n], "area": NM[wc], "person_days": len(g), "first_year": int(g.date.dt.year.min()), "last_year": int(g.date.dt.year.max()), "labour_hrs": g.hrs.sum(), "std_hrs_per_labour_hr": g.earned.sum() / g.hrs.sum(), "share_of_area_bookings_pct": 100 * g.n.sum() / fo[fo.work_centre == wc].n.sum(), "days_2026": len(g26), "std_hrs_per_labour_hr_2026": g26.earned.sum() / g26.hrs.sum() if g26.hrs.sum() > 0 else None})
save("operator_table", sorted(opr, key=lambda r: (r["area"], -r["labour_hrs"])), "Operators with 60+ days in an area, names replaced by codes. Ratio above 1 means more standard hours booked than attendance hours, which happens when one person books for a crew.")
sp = pd.DataFrame(opr)
save("operator_spread", [{"area": a, "operators": len(g), "lowest": g.std_hrs_per_labour_hr.min(), "median": g.std_hrs_per_labour_hr.median(), "highest": g.std_hrs_per_labour_hr.max(), "top_booker_share_pct": g.share_of_area_bookings_pct.max(), "active_2026": int((g.days_2026 > 0).sum())} for a, g in sp.groupby("area")], "Spread of booked output per attendance hour across operators in each area.")
print("report3 done", flush=True)
