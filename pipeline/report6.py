"""PCT: casting weights and pattern sizes; Core Engineering planning schedule.

Usage: python report6.py <wip2.parquet> <report_dir> <ext_dir>
Reads PCT_Pattern_Weights.xlsx and PCT_Core_Schedule.xlsx. Writes JSON with prefix w_ (weights) and k_ (schedule).
"""
import sys, os, json, math, re, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
SRC, OUT, EXT = sys.argv[1:4]
CODE = open(os.path.abspath(__file__)).read()
def clean(v):
    if v is pd.NaT: return None
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Period): return str(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    json.dump({"rows": rows, "meta": [{"label": "report6.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]}, open(f"{OUT}/{name}.json", "w"), allow_nan=False); print("wrote " + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]
norm = lambda s: re.sub(r"\s+", "", str(s).upper())
TODAY = pd.Timestamp("2026-11-20")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["key"] = d.part_no.map(norm); WEND = d.date.max(); ann = 365 / WEND.dayofyear
mo = d[d.op.between(10, 19)]; ck = set(mo.key)
# ================= weights
p = pd.read_excel(glob.glob(f"{EXT}/PCT_Pattern_Weights*.xlsx")[-1]); p["k0"] = p.part_no.map(norm)
desc_keys = {}
for k_, g in d[d.key.isin(ck)].groupby("key"): pass
def find(k):
    for c in (k, k + "C", k + "-C", k.rstrip("C"), k.replace("-C", "C")):
        if c in ck: return c
    cand = [x for x in ck if x.startswith(k) or k.startswith(x)]
    return cand[0] if len(cand) == 1 else None
p["key"] = p.k0.map(find)
my = mo.groupby(["key", mo.date.dt.year]).qty.sum().unstack(fill_value=0)
for y in range(2021, 2027): p[f"cast_{y}"] = p.key.map(my[y]) if y in my else 0
p["cast_all"] = p.key.map(mo.groupby("key").qty.sum()); p["line"] = p.key.map(mo.groupby("key").work_centre.agg(lambda s: s.mode().iat[0]))
cy = {int(r["year"]): r for r in J("x_commercial_year")}; kg26 = cy[2026]["al_gbp_per_kg"]; rate26 = cy[2026]["loaded_min_rate"]
rp = {norm(r["part_no"]): r for r in J("z_reprice_all")}; p["labour_hrs"] = p.key.map(lambda k: rp.get(k, {}).get("hrs_per_casting")); p["labour_gbp"] = p.labour_hrs * rate26
so = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Sales_Orders_*.xlsx"))[-1], sheet_name="Sales order lines", dtype={"part_no": str}); so["key"] = so.part_no.map(norm); so = so[(so.status != "Cancelled") & (so.ordered > 0) & (so.order_net > 0)].sort_values("entered")
lp = so.groupby("key").apply(lambda g: g.order_net.iloc[-3:].sum() / g.ordered.iloc[-3:].sum())
def price(r):
    for c in (r.key, r.k0, (r.key or "").rstrip("C").rstrip("-")):
        if c and c in lp.index: return lp[c]
    return np.nan
p["price"] = p.apply(price, axis=1); p["metal_gbp"] = p.weight_kg * kg26; p["price_per_kg"] = p.price / p.weight_kg; p["metal_pct_of_price"] = 100 * p.metal_gbp / p.price; p["labour_pct_of_price"] = 100 * p.labour_gbp / p.price
p["metal_to_labour"] = p.metal_gbp / p.labour_gbp; p["mould_litres"] = p.mould_breadth_mm * p.mould_height_mm * p.mould_length_mm / 1e6; p["kg_per_100_litres_mould"] = 100 * p.weight_kg / p.mould_litres
p["kg_2026"] = p.weight_kg * p.cast_2026.fillna(0) * ann; p["metal_value_2026"] = p.kg_2026 * kg26; p["hrs_per_kg"] = p.labour_hrs / p.weight_kg
p["metal_change_since_2021"] = p.weight_kg * (kg26 - cy[2021]["al_gbp_per_kg"]) * p.cast_2026.fillna(0) * ann
save("w_parts", p.sort_values("kg_2026", ascending=False).drop(columns=["k0"]), "Pattern register weights and mould sizes joined to castings moulded, foundry labour per casting (attendance model, 2026 wage floor), recent selling price and LME metal at the 2026 average. Metal is finished casting weight x LME; melt loss, runners, alloy premium and returns are not included.")
m = p[p.key.notna()]; tot26 = mo[mo.date.dt.year == 2026].qty.sum(); stdh = {}
yrs = [{"year": str(y), "castings_weighed_parts": m[f"cast_{y}"].sum(), "all_castings": mo[mo.date.dt.year == y].qty.sum(), "coverage_pct": 100 * m[f"cast_{y}"].sum() / mo[mo.date.dt.year == y].qty.sum(), "tonnes_weighed_parts": (m.weight_kg * m[f"cast_{y}"]).sum() / 1000 * (ann if y == 2026 else 1), "avg_kg_per_casting": (m.weight_kg * m[f"cast_{y}"]).sum() / max(m[f"cast_{y}"].sum(), 1), "metal_value_at_year_lme": (m.weight_kg * m[f"cast_{y}"]).sum() * cy[y]["al_gbp_per_kg"] * (ann if y == 2026 else 1)} for y in range(2021, 2027)]
save("w_year", yrs, "Castings, tonnes and metal value for the parts with a recorded weight, by year moulded. 2026 tonnes and value annualised. Coverage = share of all castings moulded that year.")
v = m.dropna(subset=["labour_hrs"]); rho, pv = stats.spearmanr(v.weight_kg, v.labour_hrs) if len(v) > 5 else (np.nan, np.nan); r2, p2 = stats.spearmanr(v.mould_litres, v.labour_hrs) if len(v) > 5 else (np.nan, np.nan)
pr = m.dropna(subset=["price"]); 
save("w_summary", [{"patterns": len(p), "matched_to_bookings": int(p.key.notna().sum()), "weight_min": p.weight_kg.min(), "weight_median": p.weight_kg.median(), "weight_max": p.weight_kg.max(), "coverage_2026_pct": yrs[-1]["coverage_pct"], "coverage_2023_pct": yrs[2]["coverage_pct"], "tonnes_2026": yrs[-1]["tonnes_weighed_parts"], "avg_kg_2026": yrs[-1]["avg_kg_per_casting"], "avg_kg_2021": yrs[0]["avg_kg_per_casting"], "metal_value_2026": yrs[-1]["metal_value_at_year_lme"],
    "rho_weight_vs_labour": rho, "p_weight_vs_labour": pv, "n_weight_vs_labour": len(v), "rho_mould_volume_vs_labour": r2, "priced_parts": len(pr), "metal_pct_of_price_median": pr.metal_pct_of_price.median(), "labour_pct_of_price_median": pr.labour_pct_of_price.median(), "price_per_kg_median": pr.price_per_kg.median(), "price_per_kg_min": pr.price_per_kg.min(), "price_per_kg_max": pr.price_per_kg.max(), "metal_to_labour_median": v.metal_to_labour.median(), "parts_metal_above_labour": int((v.metal_to_labour > 1).sum()), "parts_with_labour": len(v),
    "metal_cost_rise_since_2021_on_2026_volume": m.metal_change_since_2021.sum(), "lme_gbp_kg_2026": kg26, "lme_gbp_kg_2021": cy[2021]["al_gbp_per_kg"], "hrs_per_kg_median": v.hrs_per_kg.median(), "implied_tonnes_all_2026": yrs[-1]["tonnes_weighed_parts"] / (yrs[-1]["coverage_pct"] / 100) if yrs[-1]["coverage_pct"] else None}], "Summary of the pattern weight file. Implied tonnes scales the weighed parts to all castings and assumes the unweighed parts are of similar weight, which is unlikely to hold exactly.")
# ================= Core schedule
X = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Schedule*.xlsx")[-1], sheet_name=None); N, Nd, O, Od = X["Schedule Oct-Dec 2026"], X["Labour Oct-Dec 2026"], X["Schedule to Oct 2026"], X["Labour to Oct 2026"]
for t in (N, Nd, O, Od): t["date"] = pd.to_datetime(t.date)
wk = lambda t: t[t.date.dt.dayofweek < 5]
def mach(L, a=None, b=None):
    L = wk(L); L = L[(L.date >= a) if a is not None else slice(None)] if a is not None else L; L = L[L.date <= b] if b is not None else L; out = []
    for mname, g in L.groupby("machine"):
        e = g.entry.fillna(""); su = e.str.upper().str.contains("SETUP"); job = (e != "") & ~su
        out.append({"machine": mname, "slots": len(g), "staffed_slots": int((g.labour > 0).sum()), "staffed_pct": 100 * (g.labour > 0).mean(), "job_entries": int(job.sum()), "distinct_jobs": int(e[job].nunique()), "setups": int(su.sum()), "setups_per_week": su.sum() / max(g.date.dt.to_period("W").nunique(), 1), "setup_share_of_staffed_pct": 100 * su.sum() / max((g.labour > 0).sum(), 1)})
    return out
fwd = mach(N, TODAY, None); allp = mach(N)
save("k_sched_machine", [{**a, "staffed_pct_from_20_nov": next((f["staffed_pct"] for f in fwd if f["machine"] == a["machine"]), None), "setups_from_20_nov": next((f["setups"] for f in fwd if f["machine"] == a["machine"]), None)} for a in sorted(allp, key=lambda r: -r["staffed_slots"])], "Core Engineering planning schedule, 5 Oct to 24 Dec 2026, weekdays. Nine slots a day per machine. Staffed = a person is allocated to the machine in that slot. Setups = slots marked SETUP.")
nd = wk(Nd); save("k_sched_labour", [{"month": str(mth), "slots": len(g), "avg_people_per_slot": g.labour.mean(), "avg_free_people": g.free_labour.mean(), "slots_short_pct": 100 * (g.free_labour < 0).mean(), "slots_with_spare_pct": 100 * (g.free_labour > 0).mean(), "worst_shortfall": g.free_labour.min()} for mth, g in nd.groupby(nd.date.dt.to_period("M"))], "Labour on the Core schedule by month: people allocated per slot and free labour (negative = more machines planned than people).")
oq = wk(O[O.date >= "2025-07-01"]); q = oq.groupby([oq.date.dt.to_period("Q"), "machine"]).labour.apply(lambda s: 100 * (s > 0).mean()).unstack().fillna(0); q = q.loc[:, q.max() > 3]
save("k_sched_history", [{"quarter": str(i), **{c: r[c] for c in q.columns}} for i, r in q.iterrows()], "Share of weekday schedule slots with a person allocated, by machine and quarter, from the earlier schedule (to 4 Oct 2026).")
# schedule against the order book and Continuum
N["wo"] = N.entry.fillna("").str.findall(r"WO\s?(\d{5})"); sx = N.explode("wo").dropna(subset=["wo"]); sw = sx.groupby("wo").agg(first=("date", "min"), last=("date", "max"), machine=("machine", "first"), slots=("date", "size"))
ob = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Order_Book_*.xlsx"))[-1], dtype={"works_orders": str, "part_no": str, "so": str}); ob["wos"] = ob.works_orders.fillna("").str.findall(r"\d{5}"); ob = ob[~ob.part_no.str.contains(r"CARRIAGE|C OF C|CHARGE", case=False)]; ob["required"] = pd.to_datetime(ob.required)
c = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Works_Orders_Continuum*.xlsx")[-1], sheet_name="Tasks", dtype={"works_order": str}); c["wos"] = c.works_order.fillna("").str.findall(r"\d{5}"); cx = c.explode("wos"); cfin = set(cx[cx.status == "Finished"].wos.dropna()); clive = set(cx[cx.status.isin(["Scheduled", "Started", "Unscheduled"])].wos.dropna())
rows = []
for r in ob.itertuples():
    s = sw.reindex(r.wos).dropna(how="all"); live = any(w in clive for w in r.wos); fin = any(w in cfin for w in r.wos) and not live
    rows.append({"customer": r.customer, "part_no": r.part_no, "so": r.so, "required": r.required, "value": r.os_net, "units": r.os_qty, "works_orders": r.works_orders, "on_schedule": len(s) > 0, "sched_first": s["first"].min() if len(s) else pd.NaT, "sched_last": s["last"].max() if len(s) else pd.NaT, "machine": s.machine.iloc[0] if len(s) else None, "continuum_live": live, "continuum_finished": fin})
b = pd.DataFrame(rows); b["slack_days"] = (b.required - b.sched_last).dt.days; due = b[b.required <= "2026-12-24"]
save("k_sched_vs_book", [{"open_product_lines": len(b), "value": b.value.sum(), "lines_on_schedule": int(b.on_schedule.sum()), "value_on_schedule": b[b.on_schedule].value.sum(), "scheduled_to_finish_after_required_lines": int((b.slack_days < 0).sum()), "scheduled_to_finish_after_required_value": b[b.slack_days < 0].value.sum(), "finish_within_7_days_of_required_lines": int(((b.slack_days >= 0) & (b.slack_days < 7)).sum()), "finish_within_7_days_value": b[(b.slack_days >= 0) & (b.slack_days < 7)].value.sum(), "comfortable_lines": int((b.slack_days >= 7).sum()), "comfortable_value": b[b.slack_days >= 7].value.sum(),
    "due_by_24_dec_lines": len(due), "due_by_24_dec_value": due.value.sum(), "due_not_on_schedule_not_finished_lines": int((~due.on_schedule & ~due.continuum_finished).sum()), "due_not_on_schedule_not_finished_value": due[~due.on_schedule & ~due.continuum_finished].value.sum(), "due_not_on_schedule_but_live_in_continuum_value": due[~due.on_schedule & ~due.continuum_finished & due.continuum_live].value.sum(), "schedule_wos": len(sw), "schedule_wos_in_continuum_pct": 100 * pd.Series(sw.index).isin(set(cx.wos.dropna())).mean(), "schedule_entries_without_wo_pct": 100 * (N.entry.notna() & ~N.entry.fillna("").str.upper().str.contains("SETUP") & (N.wo.str.len() == 0)).sum() / max((N.entry.notna() & ~N.entry.fillna("").str.upper().str.contains("SETUP")).sum(), 1)}],
     "Open order lines (charges left out) against the Core planning schedule to 24 Dec 2026, joined on works order numbers written in schedule cells. A line counts as on the schedule only if one of its works order numbers appears in a cell; jobs entered by name alone cannot be matched.")
late = b[(b.slack_days < 7) | (~b.on_schedule & ~b.continuum_finished & (b.required <= "2026-12-24"))].sort_values("value", ascending=False)
save("k_sched_exposed", [{"customer": r.customer, "part_no": r.part_no, "so": r.so, "required": r.required, "value": r.value, "units": r.units, "works_orders": r.works_orders, "machine": r.machine, "scheduled_last": r.sched_last, "slack_days": r.slack_days, "position": ("Scheduled after required" if r.slack_days < 0 else "Under a week of slack") if r.on_schedule else ("Not on schedule; live in Continuum" if r.continuum_live else "Not on schedule or in Continuum")} for r in late.head(20).itertuples()], "Open lines due by 24 Dec 2026 that are scheduled late, scheduled with under a week of slack, or not found on the schedule.")
print("report6 done", flush=True)
