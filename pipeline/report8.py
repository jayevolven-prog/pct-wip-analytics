"""PCT: Core Engineering extended analytics, matching the depth of the foundry analysis.

Usage: python report8.py <wip2.parquet> <report_dir> <ext_dir>
Writes JSON with prefix e_.
"""
import sys, os, json, math, re, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
SRC, OUT, EXT = sys.argv[1:4]
CODE = open(os.path.abspath(__file__)).read()
def clean(v):
    if v is pd.NaT or v is None: return None
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Period): return str(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    json.dump({"rows": rows, "meta": [{"label": "report8.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]}, open(f"{OUT}/{name}.json", "w"), allow_nan=False); print("wrote " + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]
norm = lambda s: re.sub(r"\s+", "", str(s).upper())
TODAY = pd.Timestamp("2026-11-20"); T0 = pd.Timestamp("2025-07-01")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["wo"] = d.wo_base.astype(str); d["key"] = d.part_no.map(norm)
c = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Works_Orders_Continuum*.xlsx")[-1], sheet_name="Tasks", dtype={"works_order": str}); c["wos"] = c.works_order.fillna("").str.findall(r"\d{5}"); c["key"] = c.part_no.map(norm)
so = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Sales_Orders_*.xlsx"))[-1], sheet_name="Sales order lines", dtype={"part_no": str}); so["key"] = so.part_no.map(norm); sv = so[(so.status != "Cancelled") & (so.ordered > 0) & (so.order_net > 0)].sort_values("entered")
lp = sv.groupby("key").apply(lambda g: g.order_net.iloc[-3:].sum() / g.ordered.iloc[-3:].sum()); cust = sv.groupby("key").customer.last()
ob = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Order_Book_*.xlsx"))[-1], dtype={"works_orders": str, "part_no": str}); obx = ob.assign(wos=ob.works_orders.fillna("").str.findall(r"\d{5}")).explode("wos"); wo_part = obx.dropna(subset=["wos"]).groupby("wos").part_no.first().map(norm); wo_cust = obx.dropna(subset=["wos"]).groupby("wos").customer.first()
pk = set(lp.index)
def sold_key(r):
    k = r.key
    for cand in (k, k.replace("PRE-ZX8-", ""), re.sub(r"^PRE-", "", k)):
        if cand in pk: return cand
    for w in r.wos:
        if w in wo_part.index and wo_part[w] in pk: return wo_part[w]
        for nb in (str(int(w) + 1), str(int(w) - 1)):
            if nb in wo_part.index and wo_part[nb] in pk: return wo_part[nb]
    m = [x for x in pk if len(k) >= 6 and (x.startswith(k) or k.startswith(x))]
    return m[0] if len(m) == 1 else None
c["skey"] = c.apply(sold_key, axis=1); c["price"] = c.skey.map(lp); c["customer"] = c.skey.map(cust)
def cust_from_desc(r):
    if isinstance(r.customer, str): return r.customer
    t = f"{r.task} {r.description}".upper()
    for k, v in (("STREET", "STREET CRANE COMPANY LIMITED"), ("ZX", "STREET CRANE COMPANY LIMITED"), ("JCE", "JCE"), ("ALSTOM", "ALSTOM ENGINEERING AND SERVICES LIMITED"), ("ALSTROM", "ALSTOM ENGINEERING AND SERVICES LIMITED"), ("SSS", "SSS GEARS LTD"), ("KINETROL", "KINETROL LIMITED"), ("ROTORK", "ROTORK CONTROLS LIMITED"), ("PAK BOX", "ROTORK CONTROLS LIMITED"), ("AIR CONTROL", "AIR CONTROL INDUSTRIES LIMITED"), ("FANCASE", "AIR CONTROL INDUSTRIES LIMITED"), ("PANDROL", "PANDROL LIMITED"), ("HERCULES", "HERCULES ATLANTIC PRODUCTS LIMITED"), ("URBIS", "URBIS SCHREDER LTD"), ("CALMFLOOR", "ENGINEERED VIBRATION SOLUTIONS LTD")):
        if k in t: return v
    return "Other or not identified"
c["customer"] = c.apply(cust_from_desc, axis=1)
fin = c[c.status == "Finished"].copy(); fin["when"] = fin.finish.fillna(fin.start); fin["month"] = fin.when.dt.to_period("M"); fin["q"] = fin.when.dt.to_period("Q"); fin["good"] = np.minimum(fin.current_qty, fin.expected_qty); fin["short"] = (fin.expected_qty - fin.current_qty).clip(lower=0); fin["value"] = fin.current_qty * fin.price
m26 = TODAY.dayofyear / 30.4375
# ---- A. customers and parts
tot_u = fin.current_qty.sum(); priced_u = fin[fin.price.notna()].current_qty.sum()
cg = fin.groupby("customer").agg(tasks=("task_id", "size"), parts=("part_no", "nunique"), units=("current_qty", "sum"), value=("value", "sum"), units_2026=("current_qty", lambda s: s[fin.loc[s.index, "when"] >= "2026-01-01"].sum())).reset_index().sort_values("units", ascending=False)
cg["units_share_pct"] = 100 * cg.units / tot_u; cg["value_share_pct"] = 100 * cg.value / cg.value.sum(); cg["units_per_month_2026"] = cg.units_2026 / m26
save("e_core_customers", cg, "Core Engineering output by customer. Customer comes from the sales order for the part, else from the task or part description. Value = units finished x the part's recent selling price, where the part can be matched to a sales order line; it is the sales value of the finished part, not the machining charge.")
pg = fin.groupby("part_no").agg(description=("description", "first"), customer=("customer", "first"), tasks=("task_id", "size"), units=("current_qty", "sum"), expected=("expected_qty", "sum"), price=("price", "first"), value=("value", "sum"), machines=("machine", lambda s: ", ".join(sorted(set(s) - {"Unassigned"}))[:36]), median_batch=("expected_qty", "median")).reset_index().sort_values("units", ascending=False)
pg["cum_units_pct"] = 100 * pg.units.cumsum() / pg.units.sum(); pg["short"] = (pg.expected - pg.units).clip(lower=0); pg["short_pct"] = 100 * pg.short / pg.expected.replace(0, np.nan); pg["short_value"] = pg.short * pg.price
save("e_core_pareto", pg.head(25), "Parts machined at Core Engineering, ranked by units finished, with cumulative share.")
sh = 100 * pg.units / pg.units.sum(); shc = cg.units_share_pct
save("e_core_mix", [{"parts": len(pg), "top5_parts_share_pct": sh.head(5).sum(), "top10_parts_share_pct": sh.head(10).sum(), "parts_for_80pct": int((pg.cum_units_pct < 80).sum() + 1), "hhi_parts": (sh ** 2).sum(), "customers": int((cg.customer != "Other or not identified").sum()), "top_customer": cg.customer.iloc[0], "top_customer_share_pct": shc.iloc[0], "top3_customer_share_pct": shc.head(3).sum(), "hhi_customers": (shc ** 2).sum(), "one_off_parts": int((pg.tasks == 1).sum()), "one_off_parts_pct": 100 * (pg.tasks == 1).mean(), "median_batch": fin.expected_qty.median(), "batch_p25": fin.expected_qty.quantile(.25), "batch_p75": fin.expected_qty.quantile(.75), "batches_under_10_pct": 100 * (fin.expected_qty < 10).mean(), "priced_units_pct": 100 * priced_u / tot_u, "value_machined": fin.value.sum(), "value_machined_2026": fin[fin.when >= "2026-01-01"].value.sum(), "value_per_month_2026": fin[fin.when >= "2026-01-01"].value.sum() / m26}], "Core Engineering product and customer mix. HHI above 2,500 is highly concentrated.")
# ---- B. shortfall (yield proxy)
sp = pg[pg.short > 0].sort_values("short_value", ascending=False, na_position="last")
save("e_core_short_parts", pd.concat([sp[sp.short_value.notna()].head(12), sp.sort_values("short", ascending=False).head(12)]).drop_duplicates("part_no").head(15)[["part_no", "description", "customer", "tasks", "expected", "units", "short", "short_pct", "price", "short_value"]], "Parts finishing short of the planned quantity at Core Engineering: units expected less units finished on finished tasks. Short value = shortfall x recent selling price. A shortfall may be scrap, a part-shipped batch or a task closed early; Continuum does not say which.")
sm = fin.groupby("month").agg(expected=("expected_qty", "sum"), finished=("current_qty", "sum"), good=("good", "sum"), short=("short", "sum"), tasks=("task_id", "size"), tasks_short=("short", lambda s: (s > 0).sum()), zero=("current_qty", lambda s: (s == 0).sum())).reset_index()
base = sm[sm.month < "2026-07"]; pbar = base.short.sum() / base.expected.sum(); sm["short_pct"] = 100 * sm.short / sm.expected; sm["centre_pct"] = 100 * pbar
bt = fin[fin.month < "2026-07"]; phi = math.sqrt(max(1.0, ((bt.short / bt.expected_qty.replace(0, np.nan)).var() * bt.expected_qty.mean()) / (pbar * (1 - pbar))))
sm["ucl_pct"] = 100 * (pbar + 3 * phi * np.sqrt(pbar * (1 - pbar) / sm.expected)); sm["signal"] = np.where(sm.short_pct > sm.ucl_pct, "above limit", ""); sm["month"] = sm.month.astype(str)
save("e_core_short_month", sm, "Monthly shortfall at Core Engineering (units expected less finished, finished tasks), with a Laney-adjusted p-chart limit from tasks finished before July 2026.")
ms = fin.groupby("machine").agg(tasks=("task_id", "size"), expected=("expected_qty", "sum"), short=("short", "sum"), tasks_short=("short", lambda s: (s > 0).sum()), zero_qty=("current_qty", lambda s: (s == 0).sum()), short_value=("short", lambda s: (s * fin.loc[s.index, "price"]).sum())).reset_index(); ms["short_pct"] = 100 * ms.short / ms.expected; ms["tasks_short_pct"] = 100 * ms.tasks_short / ms.tasks
save("e_core_short_machine", ms.sort_values("short", ascending=False), "Shortfall by Core machine.")
save("e_core_short_summary", [{"expected": fin.expected_qty.sum(), "finished": fin.current_qty.sum(), "short": fin.short.sum(), "short_pct": 100 * fin.short.sum() / fin.expected_qty.sum(), "over": (fin.current_qty - fin.expected_qty).clip(lower=0).sum(), "tasks_short_pct": 100 * (fin.short > 0).mean(), "zero_qty_tasks": int((fin.current_qty == 0).sum()), "short_excl_zero": fin[fin.current_qty > 0].short.sum(), "short_pct_excl_zero": 100 * fin[fin.current_qty > 0].short.sum() / fin[fin.current_qty > 0].expected_qty.sum(), "short_value_priced": (fin.short * fin.price).sum(), "short_value_excl_zero": (fin[fin.current_qty > 0].short * fin[fin.current_qty > 0].price).sum(), "pct_scrap_report_cnc_milling_pct": 1.59, "pct_scrap_report_turning_pct": 2.11}], "Shortfall totals. Tasks finished with zero quantity are likely closed without a count, so the rate is shown with and without them. The last two fields are the PCT Scrap Analysis rates for CNC milling and turning, 2021 to 2026, for comparison.")
# ---- C. flow through Core
r = fin[((fin.finish - fin.start).dt.total_seconds() / 3600) > 1].copy(); r["days"] = (r.finish - r.start).dt.total_seconds() / 86400; r["per_unit_hr"] = r.days * 24 / r.current_qty.replace(0, np.nan)
save("e_core_flow_machine", [{"machine": m, "tasks_with_duration": len(g), "share_of_finished_pct": 100 * len(g) / max((fin.machine == m).sum(), 1), "median_days": g.days.median(), "p75_days": g.days.quantile(.75), "p90_days": g.days.quantile(.9), "median_batch": g.expected_qty.median(), "median_elapsed_hrs_per_unit": g.per_unit_hr.median()} for m, g in r.groupby("machine") if len(g) >= 5], "Elapsed days from task start to finish at Core Engineering, for finished tasks that recorded a real duration (over one hour). Elapsed time includes nights, weekends and waiting, so it is a lead time, not a cycle time.")
save("e_core_flow_quarter", [{"quarter": str(q), "tasks_with_duration": len(g), "median_days": g.days.median(), "p75_days": g.days.quantile(.75), "p90_days": g.days.quantile(.9)} for q, g in r.groupby("q") if len(g) >= 5], "The same by quarter finished.")
bs = pd.cut(r.expected_qty, [0, 9, 24, 49, 9999], labels=["1 to 9", "10 to 24", "25 to 49", "50 and over"])
save("e_core_flow_batch", [{"batch": str(b), "tasks": len(g), "median_days": g.days.median(), "p75_days": g.days.quantile(.75), "median_elapsed_hrs_per_unit": g.per_unit_hr.median()} for b, g in r.groupby(bs) if len(g) >= 5], "Elapsed days by batch size.")
# ---- D. schedule: calendar, labour, correlations, control chart
X = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Schedule*.xlsx")[-1], sheet_name=None); N, O, Nd, Od = X["Schedule Oct-Dec 2026"], X["Schedule to Oct 2026"], X["Labour Oct-Dec 2026"], X["Labour to Oct 2026"]
N["hrs"] = 8.5 / 9; O["hrs"] = 8.5 / 4; Nd["hrs"] = 8.5 / 9; Od["hrs"] = 8.5 / 4; S = pd.concat([O, N], ignore_index=True); S["date"] = pd.to_datetime(S.date); L = pd.concat([Od, Nd], ignore_index=True); L["date"] = pd.to_datetime(L.date)
MACH = ["Mazak HCN5000", "Hedelius T11", "YCM", "AKARI", "Doosan", "Smart Turn 300", "XYZ", "VTC"]; STOPS = {"Forced stop", "Awaiting material or tooling", "Awaiting castings or material", "Requires programming"}; RUN = {"Firm order", "Work order complete", "New job", "Planned", "Mould", "CMM inspection", "At powder coating"}
def cat(x):
    if x.status == "Setup": return "Setup"
    if x.status in STOPS: return "Stop"
    if x.status == "Audit, risk or training": return "Other"
    if x.status in RUN or (isinstance(x.entry, str) and x.entry.strip()) or x.labour > 0: return "Run"
    return "Idle"
A = S[(S.date >= T0) & (S.date < TODAY) & S.machine.isin(MACH) & (S.date.dt.dayofweek < 5) & ~S.status.isin(["Weekend or bank holiday", "Holiday"])].copy(); A["cat"] = A.apply(cat, axis=1); A["job"] = A.entry.fillna("").str.split("|").str[0].str.strip().str.upper()
dn = ["Mon", "Tue", "Wed", "Thu", "Fri"]; wdr = []
for i in range(5):
    g = A[A.date.dt.dayofweek == i]; h = g.groupby("cat").hrs.sum(); ld = h.sum() - h.get("Idle", 0); f = fin[fin.when.dt.dayofweek == i]; nd_ = g.date.nunique()
    wdr.append({"weekday": dn[i], "days": nd_, "loaded_hrs_per_day": ld / nd_, "run_hrs_per_day": h.get("Run", 0) / nd_, "uptime_pct": 100 * h.get("Run", 0) / ld, "setup_pct": 100 * h.get("Setup", 0) / ld, "stopped_pct": 100 * h.get("Stop", 0) / ld, "tasks_finished": len(f), "units_finished": f.current_qty.sum(), "share_of_units_pct": 100 * f.current_qty.sum() / fin.current_qty.sum()})
save("e_core_weekday", wdr, "Core Engineering by weekday: loaded hours and uptime from the schedule, tasks and units finished from Continuum.")
Lw = L[(L.date >= T0) & (L.date < TODAY) & (L.date.dt.dayofweek < 5)].copy(); Lw["month"] = Lw.date.dt.to_period("M"); Lw["ph"] = Lw.labour.fillna(0) * Lw.hrs
A["month"] = A.date.dt.to_period("M"); mrows = []
for mth, g in A.groupby("month"):
    h = g.groupby("cat").hrs.sum(); ld = h.sum() - h.get("Idle", 0); f = fin[fin.month == mth]; lw = Lw[Lw.month == mth]; ph = lw.ph.sum(); su = g[g.cat == "Setup"]
    mrows.append({"month": str(mth), "Loaded machine hours": ld, "Run hours": h.get("Run", 0), "Uptime %": 100 * h.get("Run", 0) / ld if ld else None, "Setup hours": h.get("Setup", 0), "Stopped hours": h.get("Stop", 0), "Units finished": f.current_qty.sum(), "Tasks finished": len(f), "Different jobs run": g[(g.cat == "Run") & (g.job != "")].job.nunique(), "Machines in use": g[g.cat != "Idle"].machine.nunique(), "Night hours": g[(g["shift"] == "night") & (g.cat != "Idle")].hrs.sum(), "People per slot": lw.labour.mean(), "Person hours": ph, "Slots short of people %": 100 * (lw.free_labour < 0).mean() if len(lw) else None, "Shortfall %": 100 * f.short.sum() / f.expected_qty.sum() if f.expected_qty.sum() else None, "Median batch": f.expected_qty.median(), "Units per run hour": f.current_qty.sum() / h.get("Run", 0) if h.get("Run", 0) > 50 else None, "Units per person hour": f.current_qty.sum() / ph if ph > 50 else None, "Machine hours per person hour": ld / ph if ph > 50 else None, "Sales value machined": f.value.sum()})
M = pd.DataFrame(mrows); save("e_core_monthly", M, "Core Engineering monthly series: schedule hours, labour, Continuum output and sales value of parts finished.")
cols = ["Units finished", "Run hours", "Loaded machine hours", "Person hours", "Uptime %", "Setup hours", "Stopped hours", "Different jobs run", "Night hours", "Slots short of people %", "Shortfall %", "Median batch", "Units per run hour"]; Mc = M[M.month >= "2025-09"][cols]; pr = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        x = Mc[[cols[i], cols[j]]].dropna()
        if len(x) < 10 or x.iloc[:, 0].nunique() < 4 or x.iloc[:, 1].nunique() < 4: continue
        rho, p_ = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1]); pr.append({"a": cols[i], "b": cols[j], "rho": rho, "p": p_, "n": len(x)})
ps = np.array([x["p"] for x in pr]); od = np.argsort(ps); adj = np.empty(len(ps)); prev = 1.0
for rank, idx in list(enumerate(od, 1))[::-1]: prev = min(prev, ps[idx] * len(ps) / rank); adj[idx] = prev
for x, a_ in zip(pr, adj): x["p_fdr"] = a_; x["significant"] = bool(a_ < 0.05)
save("e_core_corr", sorted(pr, key=lambda x: -abs(x["rho"])), f"Spearman rank correlation between Core Engineering monthly series, September 2025 to November 2026 ({len(Mc)} months), Benjamini-Hochberg 5%. Few months, so only strong links can show.")
cm_ = Mc.corr(method="spearman"); save("e_core_corr_matrix", [{"a": a, "b": b_, "rho": cm_.loc[a, b_]} for a in cols for b_ in cols], "Spearman matrix, Core monthly series.")
wkf = fin[fin.when >= "2025-09-01"].set_index("when").current_qty.resample("W").sum(); wkf = wkf[wkf.index < TODAY - pd.Timedelta(days=6)]; b0 = wkf[wkf.index < "2026-07-01"]; cl = b0.mean(); mr = b0.diff().abs().mean(); ucl, lcl = cl + 2.66 * mr, max(cl - 2.66 * mr, 0)
side = np.sign(wkf - cl); run = side.groupby((side != side.shift()).cumsum()).cumcount() + 1; sig = np.where(wkf > ucl, "above limit", np.where(wkf < lcl, "below limit", np.where((run >= 8) & (side < 0), "run below average", np.where((run >= 8) & (side > 0), "run above average", ""))))
save("e_core_control", [{"week": t, "units": v, "centre": cl, "ucl": ucl, "lcl": lcl, "signal": s_} for t, v, s_ in zip(wkf.index, wkf.values, sig)], "XmR control chart of units finished per week at Core Engineering; limits from September 2025 to June 2026.")
A["week"] = A.date.dt.to_period("W"); wu = A.groupby("week").apply(lambda g: 100 * g[g.cat == "Run"].hrs.sum() / max(g[g.cat != "Idle"].hrs.sum(), 1)); wu = wu[A.groupby("week").apply(lambda g: g[g.cat != "Idle"].hrs.sum()) > 60]; b1 = wu[wu.index < pd.Period("2026-07-01", "W")]; c2 = b1.mean(); m2 = b1.diff().abs().mean()
save("e_core_control_uptime", [{"week": str(t.start_time.date()), "uptime_pct": v, "centre": c2, "ucl": min(c2 + 2.66 * m2, 100), "lcl": c2 - 2.66 * m2, "signal": "below limit" if v < c2 - 2.66 * m2 else ""} for t, v in wu.items()], "XmR control chart of weekly uptime at Core Engineering.")
# ---- E. old PCT machine shop against Core
mc = d[d.op.between(100, 199) & (d.hrs_clean > 0)]; md = d[d.work_centre == "MDESP"]; orows = []
for y in (2021, 2022, 2023, 2024, 2025):
    g = mc[mc.date.dt.year == y]; mths = 12
    orows.append({"period": f"PCT machine shop {y}", "machining_hrs_per_month": g.hrs_clean.sum() / mths, "units_out_per_month": md[md.date.dt.year == y].qty.sum() / mths, "hrs_per_unit_out": g.hrs_clean.sum() / max(md[md.date.dt.year == y].qty.sum(), 1), "work_centres": g.work_centre.nunique(), "parts": g.part_no.nunique(), "people": g.name.nunique()})
a26 = A[A.date >= "2026-01-01"]; run26 = a26[a26.cat == "Run"].hrs.sum(); f26 = fin[fin.when >= "2026-01-01"]
orows.append({"period": "Core Engineering 2026", "machining_hrs_per_month": run26 / m26, "units_out_per_month": f26.current_qty.sum() / m26, "hrs_per_unit_out": run26 / f26.current_qty.sum(), "work_centres": a26[a26.cat != "Idle"].machine.nunique(), "parts": f26.part_no.nunique(), "people": None})
save("e_core_vs_pct", orows, "PCT machine shop (timed machining hours booked, units through machined despatch) against Core Engineering 2026 (run hours from the schedule, units finished in Continuum). The bases differ: PCT hours were booked per operation by operators; Core hours are planned run slots.")
# ---- F. value per hour and capacity
qo = {r_["machine"]: r_ for r_ in J("q_core_oee")}; CM = {"Mazak HCN5000": "Maz-5000", "Hedelius T11": "HTII", "YCM": "YCM", "AKARI": "AKARI", "Doosan": "PUMA-3100Y", "Smart Turn 300": "Maz-300", "XYZ": "XYZ-425"}; vr = []
live = c[c.status.isin(["Scheduled", "Started", "Unscheduled"])]
for m, cn in CM.items():
    if m not in qo: continue
    f = fin[(fin.machine == cn) & (fin.when >= T0)]; rh = qo[m]["run_hrs"]; lv = live[live.machine == cn]; rate = qo[m]["units_per_run_hr"] or np.nan; a_ = a26[a26.machine == m]; wk_run = a_[a_.cat == "Run"].hrs.sum() / max(a_.date.dt.to_period("W").nunique(), 1)
    vr.append({"machine": m, "run_hrs": rh, "units": f.current_qty.sum(), "priced_units_pct": 100 * f[f.price.notna()].current_qty.sum() / max(f.current_qty.sum(), 1), "sales_value_machined": f.value.sum(), "sales_value_per_run_hr": f.value.sum() / rh if rh else None, "sales_value_per_loaded_hr": f.value.sum() / qo[m]["loaded_hrs"], "queued_units": lv.expected_qty.sum(), "queued_value": (lv.expected_qty * lv.price).sum(), "queue_run_hrs_at_own_rate": lv.expected_qty.sum() / rate if rate == rate else None, "run_hrs_per_week_2026": wk_run, "weeks_to_clear": (lv.expected_qty.sum() / rate) / wk_run if (rate == rate and wk_run) else None, "weeks_to_clear_at_full_day_shift": (lv.expected_qty.sum() / rate) / (42.5 * qo[m]["uptime_pct"] / 100) if rate == rate else None})
save("e_core_value_machine", vr, "Sales value of parts finished per run hour by Core machine (selling price of the finished part, not the machining charge; parts that cannot be priced are left out, so the figure is a floor), and the live queue converted to run hours at each machine's own rate.")
print("priced units %", round(100 * priced_u / tot_u, 1)); print("report8 done", flush=True)
