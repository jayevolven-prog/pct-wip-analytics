"""PCT: Precision Casting machine shop (2021 to 2025) against Core Engineering (2025 to 2026), by machine and by part.

Usage: python report9.py <wip2.parquet> <report_dir> <ext_dir>
Writes JSON with prefix v_.
"""
import sys, os, json, math, re, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SRC, OUT, EXT = sys.argv[1:4]
CODE = open(os.path.abspath(__file__)).read()
def clean(v):
    if v is pd.NaT or v is None: return None
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)): return bool(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    json.dump({"rows": rows, "meta": [{"label": "report9.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]}, open(f"{OUT}/{name}.json", "w"), allow_nan=False); print("wrote " + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]
norm = lambda s: re.sub(r"\s+", "", str(s).upper())
TODAY = pd.Timestamp("2026-11-20"); T0 = pd.Timestamp("2025-07-01")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["wo"] = d.wo_base.astype(str); d["key"] = d.part_no.map(norm)
mc = d[d.op.between(100, 199) & (d.date < "2026-01-01")].copy(); mt = mc[mc.hrs_clean > 0]
c = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Works_Orders_Continuum*.xlsx")[-1], sheet_name="Tasks", dtype={"works_order": str}); c["wos"] = c.works_order.fillna("").str.findall(r"\d{5}"); c["key"] = c.part_no.map(lambda s: norm(s).replace("PRE-ZX8-", "").replace("_B", "")); fin = c[c.status == "Finished"].copy(); fin["short"] = (fin.expected_qty - fin.current_qty).clip(lower=0)
X = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Schedule*.xlsx")[-1], sheet_name=None); N, O = X["Schedule Oct-Dec 2026"], X["Schedule to Oct 2026"]; N["hrs"] = 8.5 / 9; O["hrs"] = 8.5 / 4; S = pd.concat([O, N], ignore_index=True); S["date"] = pd.to_datetime(S.date)
S = S[(S.date >= T0) & (S.date < TODAY) & (S.date.dt.dayofweek < 5)]; RUN = {"Firm order", "Work order complete", "New job", "Planned"}
S["is_setup"] = S.status == "Setup"; S["is_run"] = (~S.is_setup) & (S.status.isin(RUN) | (S.entry.fillna("").str.strip() != "")) & ~S.status.isin(["Forced stop", "Awaiting material or tooling", "Awaiting castings or material", "Requires programming", "Audit, risk or training", "Weekend or bank holiday", "Holiday", "Mould", "CMM inspection"])
# map schedule job labels to a part key
wo_part = c.explode("wos").dropna(subset=["wos"]).groupby("wos").key.first(); ckeys = sorted(set(c.key), key=len, reverse=True)
ALIAS = {"ZX8": "220-942", "ZX6": "222-1831BO", "ZX10": "220-17272", "COPPER J'S": "916/5006DA1", "IMPEDANCE BOND": "280/119332DAO", "ET300": "254/520081556A2"}
def job_key(e):
    if not isinstance(e, str): return None
    u = e.upper(); m = re.search(r"WO\s?(\d{5})", u)
    if m and m.group(1) in wo_part.index: return wo_part[m.group(1)]
    head = norm(u.split("|")[0])
    for k in ckeys:
        if len(k) >= 5 and (head.startswith(k) or k.startswith(head) and len(head) >= 5): return k
    for a, k in ALIAS.items():
        if u.strip().startswith(a): return norm(k)
    return None
S["pkey"] = S.entry.map(job_key); S["pkey"] = S.groupby(["machine", "shift", "date"]).pkey.ffill()
core_hrs = S[S.is_run & S.pkey.notna()].groupby("pkey").hrs.sum(); core_setup = S[S.is_setup].groupby(["machine"]).hrs.sum()
cu = fin.groupby("key").agg(core_units=("current_qty", "sum"), core_expected=("expected_qty", "sum"), core_short=("short", "sum"), core_tasks=("task_id", "size"), core_machines=("machine", lambda s: ", ".join(sorted(set(s) - {"Unassigned"}))[:30]), core_batch=("expected_qty", "median"), description=("description", "first"))
# PCT hours per unit by part: timed machining hours over units through the last machining op per works order
jq = mc.groupby(["key", "wo"]).qty.max().groupby("key").sum(); ph = mt.groupby("key").agg(pct_hrs=("hrs_clean", "sum"), pct_setup_hrs=("hrs_clean", lambda s: s[mt.loc[s.index, "type"] == "Setup"].sum()), pct_orders=("wo", "nunique"), pct_centres=("work_centre", lambda s: ", ".join(s.value_counts().index[:2])), pct_first=("date", "min"), pct_last=("date", "max"))
ph["pct_units"] = jq; ph["pct_batch"] = mc.groupby(["key", "wo"]).qty.max().groupby("key").median()
pm = ph.join(cu, how="inner").join(core_hrs.rename("core_run_hrs"), how="left"); pm = pm[(pm.pct_units >= 20) & (pm.core_units >= 10)]
pm["pct_hrs_per_unit"] = pm.pct_hrs / pm.pct_units; pm["core_hrs_per_unit"] = pm.core_run_hrs / pm.core_units; pm["change_pct"] = 100 * (pm.core_hrs_per_unit / pm.pct_hrs_per_unit - 1); pm["core_short_pct"] = 100 * pm.core_short / pm.core_expected
pm["part_no"] = pm.index.map(d.groupby("key").part_no.first()); ok = pm[pm.core_run_hrs.notna() & (pm.core_run_hrs >= 8)]
save("v_part_compare", ok.sort_values("core_units", ascending=False).reset_index(drop=True)[["part_no", "description", "pct_centres", "pct_orders", "pct_units", "pct_hrs", "pct_hrs_per_unit", "pct_batch", "core_machines", "core_tasks", "core_units", "core_run_hrs", "core_hrs_per_unit", "core_batch", "change_pct", "core_short_pct"]], "Parts machined in both shops. Precision Casting: timed machining hours booked by operators (all operations, run and setup, 2021 to 2025) over units through the works order. Core Engineering: run hours planned on the schedule against that part (matched by works order number or job label) over units finished in Continuum, July 2025 to November 2026. The two time bases differ: booked operator hours can cover several operations and machines; schedule hours are machine slots. Parts with under 20 units at PCT, under 10 at Core, or under 8 matched Core hours are left out.")
wq = ok.core_units; save("v_part_summary", [{"parts_in_both": len(pm), "parts_with_core_hours": len(ok), "core_units_covered": ok.core_units.sum(), "share_of_core_units_pct": 100 * ok.core_units.sum() / fin.current_qty.sum(), "pct_hrs_per_unit_weighted": (ok.pct_hrs_per_unit * wq).sum() / wq.sum(), "core_hrs_per_unit_weighted": (ok.core_hrs_per_unit * wq).sum() / wq.sum(), "median_change_pct": ok.change_pct.median(), "parts_faster_at_core": int((ok.change_pct < -10).sum()), "parts_slower_at_core": int((ok.change_pct > 10).sum()), "parts_similar": int((ok.change_pct.abs() <= 10).sum()), "schedule_run_hrs_matched_to_a_part_pct": 100 * S[S.is_run & S.pkey.notna()].hrs.sum() / S[S.is_run].hrs.sum(), "pct_batch_median": ok.pct_batch.median(), "core_batch_median": ok.core_batch.median()}], "Summary of the like-for-like part comparison, weighted by Core units.")
# ---- machine by machine
oc = {r["work_centre"]: r for r in J("oee_centre")}; qo = {r["machine"]: r for r in J("q_core_oee")}; sw = {r["item"]: r for r in J("s_scrap_work_centre")}; sm = {r["machine"]: r for r in J("e_core_short_machine")}
PAIR = [("HCN 5000", "Mazak HCN5000", "Maz-5000", "Same machine, moved"), ("AKARI", "AKARI", "AKARI", "Same machine, moved"), ("VTC300", "VTC", "VTC300", "Same machine, moved"), ("PUMA3", "Doosan", "PUMA-3100Y", "Same type (Doosan Puma lathe)"), ("QT200", "Smart Turn 300", "Maz-300", "Same type (Mazak lathe)"), ("VC500", "YCM", "YCM", "Same role (vertical machining centre)"), ("FF510", "Hedelius T11", "HTII", "Same role (5-axis or large machining centre)")]
mths_core = (TODAY - T0).days / 30.4375; rows = []
for pw, cm, cn, rel in PAIR:
    p = oc.get(pw); q = qo.get(cm); g = mt[mt.work_centre == pw]; gm = mc[mc.work_centre == pw]; act_m = g.date.dt.to_period("M").nunique()
    f = fin[fin.machine == cn]; s_ = sw.get(pw, {}); cs = sm.get(cn, {})
    rows.append({"pct_machine": pw, "core_machine": cm, "relation": rel, "pct_hrs_per_month": g.hrs_clean.sum() / max(act_m, 1), "pct_months_active": act_m, "pct_availability": p["availability"] if p else None, "pct_performance": p["performance"] if p else None, "pct_quality": p["quality"] if p else None, "pct_oee": p["oee"] if p else None, "pct_setup_pct": 100 * p["setup_hrs"] / p["recorded_hrs"] if p else None, "pct_lost_pct": 100 * p["lost_hrs"] / p["recorded_hrs"] if p else None, "pct_parts": gm.part_no.nunique(), "pct_units_per_run_hr": gm.groupby("wo").qty.max().sum() / p["run_hrs"] if p else None, "pct_scrap_pct": s_.get("scrap_pct_of_booked"), "pct_scrap_units": s_.get("scrap_qty"),
                 "core_hrs_per_month": q["loaded_hrs"] / mths_core if q else None, "core_uptime": q["uptime_pct"] if q else None, "core_performance": q["performance_pct"] if q else None, "core_quality": q["quality_pct"] if q else None, "core_oee": q["oee_pct"] if q else None, "core_setup_pct": q["setup_pct"] if q else None, "core_stopped_pct": q["stopped_pct"] if q else None, "core_parts": f.part_no.nunique(), "core_units_per_run_hr": q["units_per_run_hr"] if q else None, "core_short_pct": cs.get("short_pct"), "core_short_units": cs.get("short"), "core_utilisation": q["utilisation_pct"] if q else None})
save("v_machine_compare", rows, "Machine by machine. Precision Casting figures are from operator time bookings, 2021 to 2025 (availability = run / recorded hours; performance against the 20th percentile time per part; quality = units despatched / started; scrap from the Scrap Analysis by work centre). Core Engineering figures are from the planning schedule and Continuum, July 2025 to November 2026 (uptime = run / loaded hours; performance against the machine's best quarter; quality and shortfall = units finished against expected). Like is compared with like only loosely: the measures share names, not definitions.")
K = {r["key"]: r["value"] for r in J("kpis")}; qt = J("q_core_total")[0]; es = J("e_core_short_summary")[0]; ev = J("e_core_vs_pct")
pall = mt; rec = pall.hrs_clean.sum(); setup_p = pall[pall.type == "Setup"].hrs_clean.sum(); lost_p = d[(d.type == "Lost") & (d.date < "2026-01-01")].hrs_clean.sum()
save("v_shop_compare", [
 {"measure": "Period", "pct": "2021 to 2025", "core": "Jul 2025 to Nov 2026", "basis": ""},
 {"measure": "Machines or work centres used", "pct": int(mt.work_centre.nunique()), "core": int(len(qo)), "basis": "With booked or scheduled time"},
 {"measure": "Machining hours a month", "pct": round(mt.hrs_clean.sum() / 60), "core": round(qt["loaded_hrs"] / mths_core), "basis": "PCT booked hours; Core loaded schedule hours"},
 {"measure": "Units out a month", "pct": round(d[(d.work_centre == "MDESP") & (d.date < "2026-01-01")].qty.sum() / 60), "core": round(fin[fin.finish.fillna(fin.start) >= T0].current_qty.sum() / mths_core), "basis": "PCT machined despatch; Core units finished"},
 {"measure": "Availability or uptime %", "pct": round(float(K["oee_availability"])), "core": round(qt["uptime_pct"]), "basis": "Run time over recorded or loaded time"},
 {"measure": "Setup share of time %", "pct": round(100 * setup_p / rec), "core": round(qt["setup_pct"]), "basis": ""},
 {"measure": "Performance %", "pct": round(float(K["oee_performance"])), "core": round(float(np.nanmean([r["performance_pct"] for r in qo.values() if r["performance_pct"]]))), "basis": "PCT against best time per part; Core against best quarter per machine"},
 {"measure": "Quality %", "pct": round(float(K["oee_quality"])), "core": round(qt["quality_pct"]), "basis": "PCT despatched over started; Core finished over expected"},
 {"measure": "OEE %", "pct": round(float(K["oee_oee"])), "core": round(float(np.nansum([r["oee_pct"] * r["run_hrs"] for r in qo.values() if r["oee_pct"]]) / np.nansum([r["run_hrs"] for r in qo.values() if r["oee_pct"]]))), "basis": "Core weighted by run hours, three main machines"},
 {"measure": "Scrap or shortfall %", "pct": 1.6, "core": round(es["short_pct_excl_zero"], 1), "basis": "PCT CNC milling scrap booked; Core units short of plan, tasks with a count"},
 {"measure": "Typical batch", "pct": round(float(mc.groupby("wo").qty.max().median())), "core": round(float(fin.expected_qty.median())), "basis": "Median works order or task quantity"},
 {"measure": "Different parts machined", "pct": int(mc.part_no.nunique()), "core": int(fin.part_no.nunique()), "basis": "PCT over five years; Core over 17 months"},
 {"measure": "Night shift", "pct": "No record", "core": "Until June 2026", "basis": ""}], "Shop level comparison.")
print("report9 done", flush=True)
