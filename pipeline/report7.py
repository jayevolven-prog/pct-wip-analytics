"""PCT: Core Engineering OEE from the planning schedule and Continuum; PCT production plan against Core's plan, the order book and run rates.

Usage: python report7.py <wip2.parquet> <report_dir> <ext_dir>
Writes JSON with prefix q_ (Core OEE) and p_ (plans).
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
    if isinstance(v, pd.Period): return str(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    json.dump({"rows": rows, "meta": [{"label": "report7.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]}, open(f"{OUT}/{name}.json", "w"), allow_nan=False); print("wrote " + name, len(rows), flush=True)
norm = lambda s: re.sub(r"\s+", "", str(s).upper())
TODAY = pd.Timestamp("2026-11-20"); T0 = pd.Timestamp("2025-07-01")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["wo"] = d.wo_base.astype(str); d["key"] = d.part_no.map(norm); WEND = d.date.max()
# ================= Core OEE
X = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Schedule*.xlsx")[-1], sheet_name=None); N, O = X["Schedule Oct-Dec 2026"], X["Schedule to Oct 2026"]
N["hrs"] = 8.5 / 9; O["hrs"] = 8.5 / 4; S = pd.concat([O, N], ignore_index=True); S["date"] = pd.to_datetime(S.date); S = S[(S.date.dt.dayofweek < 5) & (S.status != "Weekend or bank holiday") & (S.status != "Holiday")]
RUN = {"Firm order", "Work order complete", "New job", "Planned", "Mould", "CMM inspection", "At powder coating"}; STOP = {"Forced stop": "Forced stop", "Awaiting material or tooling": "Awaiting castings or material", "Awaiting castings or material": "Awaiting castings or material", "Requires programming": "Requires programming"}
def cat(r):
    if r.status == "Setup": return "Setup"
    if r.status in STOP: return STOP[r.status]
    if r.status == "Audit, risk or training": return "Audit or training"
    if r.status in RUN or (isinstance(r.entry, str) and r.entry.strip()) or r.labour > 0: return "Run"
    return "Not loaded"
S["cat"] = S.apply(cat, axis=1); S["q"] = S.date.dt.to_period("Q"); S["month"] = S.date.dt.to_period("M")
MACH = ["Mazak HCN5000", "Hedelius T11", "YCM", "AKARI", "Doosan", "Smart Turn 300", "XYZ", "VTC"]; CM = {"Mazak HCN5000": "Maz-5000", "Hedelius T11": "HTII", "YCM": "YCM", "AKARI": "AKARI", "Doosan": "PUMA-3100Y", "Smart Turn 300": "Maz-300", "XYZ": "XYZ-425", "VTC": "VTC300"}
c = pd.read_excel(glob.glob(f"{EXT}/PCT_Core_Works_Orders_Continuum*.xlsx")[-1], sheet_name="Tasks", dtype={"works_order": str}); c["wos"] = c.works_order.fillna("").str.findall(r"\d{5}")
fin = c[c.status == "Finished"].copy(); fin["when"] = fin.finish.fillna(fin.start); fin["good"] = np.minimum(fin.current_qty, fin.expected_qty)
A = S[(S.date >= T0) & (S.date < TODAY) & S.machine.isin(MACH)]; dayA = A[A["shift"] == "day"]
def block(g, gd, f):
    h = g.groupby("cat").hrs.sum(); run, setup = h.get("Run", 0), h.get("Setup", 0); stop = sum(h.get(k, 0) for k in set(STOP.values())); oth = h.get("Audit or training", 0); loaded = run + setup + stop + oth; cal = gd.hrs.sum()
    return {"calendar_hrs_day_shift": cal, "loaded_hrs": loaded, "run_hrs": run, "setup_hrs": setup, "forced_stop_hrs": h.get("Forced stop", 0), "awaiting_hrs": h.get("Awaiting castings or material", 0), "programming_hrs": h.get("Requires programming", 0), "audit_training_hrs": oth, "night_hrs": g[g["shift"] == "night"].pipe(lambda z: z[z.cat != "Not loaded"]).hrs.sum(),
            "utilisation_pct": 100 * min(loaded / cal, 1.5) if cal else None, "uptime_pct": 100 * run / loaded if loaded else None, "setup_pct": 100 * setup / loaded if loaded else None, "stopped_pct": 100 * stop / loaded if loaded else None, "tasks_finished": len(f), "units_finished": f.current_qty.sum(), "units_expected": f.expected_qty.sum(), "quality_pct": 100 * f.good.sum() / f.expected_qty.sum() if f.expected_qty.sum() else None, "units_per_run_hr": f.current_qty.sum() / run if run > 20 else None}
rows, qrows = [], []
for m in MACH:
    g, gd = A[A.machine == m], dayA[dayA.machine == m]; f = fin[(fin.machine == CM[m]) & (fin.when >= T0) & (fin.when < TODAY)]
    b = block(g, gd, f)
    if b["loaded_hrs"] < 40: continue
    qs = []
    for qq, gq in g.groupby("q"):
        fq = f[f.when.dt.to_period("Q") == qq]; bq = block(gq, gd[gd.q == qq], fq)
        if bq["loaded_hrs"] >= 40: qs.append({"machine": m, "quarter": str(qq), **bq})
    rates = [x["units_per_run_hr"] for x in qs if x["units_per_run_hr"] and x["run_hrs"] >= 250]; best = max(rates) if rates else None
    for x in qs: x["performance_pct"] = 100 * min(x["units_per_run_hr"] / best, 1) if (best and x["units_per_run_hr"]) else None; x["oee_pct"] = x["uptime_pct"] * x["performance_pct"] * x["quality_pct"] / 1e4 if (x["performance_pct"] and x["quality_pct"]) else None
    b["best_quarter_units_per_run_hr"] = best; b["performance_pct"] = 100 * min(b["units_per_run_hr"] / best, 1) if (best and b["units_per_run_hr"]) else None
    b["oee_pct"] = b["uptime_pct"] * b["performance_pct"] * b["quality_pct"] / 1e4 if (b["performance_pct"] and b["quality_pct"]) else None; b["teep_pct"] = b["oee_pct"] * min(b["utilisation_pct"], 100) / 100 if b["oee_pct"] else None
    su = g[g.cat == "Setup"]; ev = (su.sort_values(["date", "slot"]).assign(k=lambda z: (z.date.diff().dt.days.fillna(9) > 0) | (z.slot.diff().fillna(9) != 1))).k.sum()
    b["setup_events"] = int(ev); b["hrs_per_setup"] = b["setup_hrs"] / ev if ev else None; b["setups_per_week"] = ev / max(g.date.dt.to_period("W").nunique(), 1); b["distinct_jobs"] = int(g[g.cat == "Run"].entry.dropna().str.split("|").str[0].str.strip().nunique())
    rows.append({"machine": m, "continuum_name": CM[m], **b}); qrows += qs
save("q_core_oee", sorted(rows, key=lambda r: -r["run_hrs"]), "Core Engineering machine OEE, 1 Jul 2025 to 19 Nov 2026, weekdays. Time comes from the planning schedule (a plan kept up to date, not a machine log): each slot is run, setup, forced stop, awaiting castings or material, requires programming, audit or training, or not loaded. Old schedule slots are 2.1 hours (four a shift), new ones 0.94 hours (nine a shift). Uptime (availability) = run hours / loaded hours. Performance = units finished in Continuum per run hour against that machine's best quarter. Quality = units finished / units expected on finished tasks, capped at 100% per task. OEE = uptime x performance x quality. Utilisation = loaded hours / day-shift calendar hours. Continuum records no scrap and no reliable task times, so performance and quality are proxies.")
save("q_core_oee_quarter", qrows, "The same measures by machine and quarter.")
tot = block(A, dayA, fin[(fin.when >= T0) & (fin.when < TODAY)]); save("q_core_total", [tot], "All Core machines together, 1 Jul 2025 to 19 Nov 2026.")
mrows = []
for mth, g in A.groupby("month"):
    h = g.groupby("cat").hrs.sum(); f = fin[fin.when.dt.to_period("M") == mth]; loaded = sum(v for k, v in h.items() if k != "Not loaded")
    mrows.append({"month": str(mth), "run_hrs": h.get("Run", 0), "setup_hrs": h.get("Setup", 0), "forced_stop_hrs": h.get("Forced stop", 0), "awaiting_hrs": h.get("Awaiting castings or material", 0), "programming_hrs": h.get("Requires programming", 0), "audit_training_hrs": h.get("Audit or training", 0), "loaded_hrs": loaded, "uptime_pct": 100 * h.get("Run", 0) / loaded if loaded else None, "units_finished": f.current_qty.sum(), "tasks_finished": len(f), "units_per_run_hr": f.current_qty.sum() / h.get("Run", 0) if h.get("Run", 0) > 50 else None, "machines_loaded": int(g[g.cat != "Not loaded"].machine.nunique()), "night_hrs": g[(g["shift"] == "night") & (g.cat != "Not loaded")].hrs.sum()})
save("q_core_month", mrows, "Core Engineering by month: hours by state from the planning schedule and units finished from Continuum.")
jb = A[A.cat == "Run"].assign(job=lambda z: z.entry.fillna("(no label)").str.split("|").str[0].str.strip().str.upper().str[:28]).groupby("job").agg(run_hrs=("hrs", "sum"), machines=("machine", lambda s: ", ".join(sorted(set(s)))[:40]), first=("date", "min"), last=("date", "max")).sort_values("run_hrs", ascending=False).reset_index()
jb["share_pct"] = 100 * jb.run_hrs / jb.run_hrs.sum(); save("q_core_jobs", jb[jb.job != "(NO LABEL)"].head(15), "Jobs with the most planned run hours at Core Engineering, by the label written in the schedule.")
st = A[A.cat.isin(["Forced stop", "Awaiting castings or material", "Requires programming"])].assign(job=lambda z: z.entry.fillna("(no label)").str.replace("|", "/").str[:40]).groupby(["cat", "machine"]).agg(hrs=("hrs", "sum"), days=("date", "nunique"), first=("date", "min"), last=("date", "max")).reset_index().sort_values("hrs", ascending=False)
save("q_core_stops", st, "Stopped time on the Core schedule by reason and machine.")
F = S[(S.date >= TODAY) & S.machine.isin(MACH + ["Pressure test / wash / assembly"])]
save("q_core_forward", [{"machine": m, "slots": len(g), "loaded_pct": 100 * (g.cat != "Not loaded").mean(), "run_hrs": g[g.cat == "Run"].hrs.sum(), "setup_hrs": g[g.cat == "Setup"].hrs.sum(), "awaiting_castings_hrs": g[g.cat == "Awaiting castings or material"].hrs.sum(), "programming_hrs": g[g.cat == "Requires programming"].hrs.sum(), "awaiting_jobs": "; ".join(sorted(set(g[g.cat == "Awaiting castings or material"].entry.dropna().str.replace(" | ", " ", regex=False))))[:120]} for m, g in F.groupby("machine") if (g.cat != "Not loaded").any()], "Core planning schedule from 20 Nov to 24 Dec 2026 by machine, including slots already marked as awaiting castings or material.")

# ================= PCT plan
P = pd.read_excel(glob.glob(f"{EXT}/PCT_Production_Plan*.xlsx")[-1], sheet_name=None); p = P["Plan"]; sc = P["Street Crane"]; p["works_order"] = p.works_order.astype(str).str.extract(r"(\d{5})")[0]; sc["works_order"] = sc.works_order.astype(str).str.extract(r"(\d{5})")[0]; p["week"] = pd.to_datetime(p.week)
p = p[p.works_order.notna() | p.part_no.notna()].copy(); WK = pd.Timestamp("2026-11-16")
mo = d[d.op.between(10, 19)]; wline = mo.groupby("wo").work_centre.agg(lambda s: s.mode().iat[0]); p["booked_line"] = p.works_order.map(wline)
lm = p.dropna(subset=["booked_line"]).groupby("line").booked_line.agg(lambda s: s.value_counts().to_dict())
wk26 = mo[mo.date >= "2026-01-01"].assign(w=lambda z: z.date.dt.to_period("W")).groupby("w").qty.sum(); act_all = wk26.mean(); act_recent = wk26[wk26.index >= pd.Period("2026-08-03", "W")].mean()
pw = []
for (ln, w), g in p.dropna(subset=["week"]).groupby(["line", "week"]): pw.append({"line": ln, "week": w, "orders": int(g.works_order.notna().sum()), "qty_planned": g.qty_order.sum(), "qty_made": g.qty_made.sum(), "outstanding": g.outstanding.clip(lower=0).sum(), "parts": g.part_no.nunique(), "past": bool(w < WK)})
pw = pd.DataFrame(pw); save("p_plan_week", pw.sort_values(["week", "line"]), "PCT production plan by line and week commencing: quantity planned, made and outstanding.")
allw = pw.groupby("week").agg(qty_planned=("qty_planned", "sum"), qty_made=("qty_made", "sum"), outstanding=("outstanding", "sum"), orders=("orders", "sum")).reset_index(); allw["past"] = allw.week < WK
bk = mo[mo.date >= "2026-07-27"].assign(week=lambda z: z.date.dt.to_period("W").dt.start_time).groupby("week").qty.sum(); allw["castings_booked"] = allw.week.map(bk)
save("p_plan_all_week", allw, "All lines by week: plan quantity, quantity marked made on the plan, and castings actually booked at moulding in the PCT system that week.")
pl = []
for ln, g in pw.groupby("line"):
    a, f = g[g.past & (g.qty_planned > 0)], g[~g.past]
    wos = p[(p.line == ln) & p.works_order.notna()]; mk = wos[wos.week < WK]
    pl.append({"line": ln, "booked_on": ", ".join(f"{k} {v}" for k, v in sorted((lm.get(ln) or {}).items(), key=lambda x: -x[1])), "past_weeks": len(a), "past_planned": a.qty_planned.sum(), "past_made": a.qty_made.sum(), "attainment_pct": 100 * a.qty_made.sum() / a.qty_planned.sum() if a.qty_planned.sum() else None, "made_per_week": a.qty_made.sum() / max(len(a), 1), "best_week_made": a.qty_made.max() if len(a) else None, "orders_per_week": a.orders.sum() / max(len(a), 1),
               "future_weeks": len(f), "future_planned": f.qty_planned.sum(), "future_outstanding": f.outstanding.sum(), "planned_per_week": f.qty_planned.sum() / max(len(f), 1), "load_vs_run_rate_pct": 100 * (f.qty_planned.sum() / max(len(f), 1)) / (a.qty_made.sum() / max(len(a), 1)) if len(a) and len(f) and a.qty_made.sum() else None, "weeks_over_best_week": int((f.qty_planned > (a.qty_made.max() if len(a) else 0)).sum()) if len(f) else 0, "last_planned_week": g.week.max(), "orders_made_over_plan_pct": 100 * (mk.qty_made > mk.qty_order).mean() if len(mk) else None, "overmake_units": (mk.qty_made - mk.qty_order).clip(lower=0).sum(), "overmake_pct": 100 * (mk.qty_made - mk.qty_order).clip(lower=0).sum() / mk.qty_order.sum() if mk.qty_order.sum() else None})
save("p_plan_line", pl, "PCT plan by line. Past weeks are those commencing before 16 Nov 2026. Attainment = made / planned. Load vs run rate = future planned per week / past made per week. Booked on = the moulding work centres where the plan's works orders were actually booked.")
# ---- link casting works orders to Core and the order book
ob = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Order_Book_*.xlsx"))[-1], dtype={"works_orders": str, "part_no": str, "so": str}); ob["wos"] = ob.works_orders.fillna("").str.findall(r"\d{5}"); ob["required"] = pd.to_datetime(ob.required)
obx = ob.explode("wos").dropna(subset=["wos"]); ob_wo = obx.groupby("wos").agg(required=("required", "min"), value=("os_net", "sum"), ob_customer=("customer", "first"), so=("so", "first"), sold_part=("part_no", "first"), all_wos=("works_orders", "first"))
cx = c.explode("wos"); cx = cx[cx.wos.notna() & (cx.wos != "")]; core_wo = cx.groupby("wos").agg(c_start=("start", "min"), c_status=("status", lambda s: "Finished" if (s == "Finished").all() else "Live"), c_machine=("machine", "first"), c_part=("part_no", "first"))
N2 = S[S.date >= "2026-10-05"].copy(); N2["wo"] = N2.entry.fillna("").str.findall(r"WO\s?(\d{5})"); sx = N2.explode("wo").dropna(subset=["wo"]); sch_wo = sx.groupby("wo").agg(s_first=("date", "min"), s_last=("date", "max"), s_machine=("machine", "first"), s_await=("cat", lambda z: (z == "Awaiting castings or material").any()))
allp = pd.concat([p[["line", "week", "customer", "works_order", "part_no", "description", "qty_order", "qty_made", "outstanding", "promise", "notes"]], sc.assign(week=pd.NaT, customer="STREET CRANE", description=None, promise=pd.NaT, notes=sc.note)[["line", "week", "customer", "works_order", "part_no", "description", "qty_order", "qty_made", "outstanding", "promise", "notes"]]], ignore_index=True)
allp = allp[allp.works_order.notna()].drop_duplicates("works_order", keep="first"); allp["n"] = allp.works_order.astype(int)
def core_link(n):
    for k in (n, n - 1, n + 1):
        if str(k) in core_wo.index or str(k) in sch_wo.index: return str(k)
    return None
def ob_link(n):
    for k in (n, n - 1, n + 1):
        if str(k) in ob_wo.index: return str(k)
    return None
allp["core_wo"] = allp.n.map(core_link); allp["ob_wo"] = allp.n.map(ob_link)
allp = allp.join(core_wo, on="core_wo").join(sch_wo, on="core_wo").join(ob_wo, on="ob_wo")
allp["first_booking"] = allp.works_order.map(d.groupby("wo").date.min()); allp["cast_despatched"] = allp.works_order.map(d[d.op.between(80, 99)].groupby("wo").date.max())
allp["open"] = allp.outstanding.fillna(0) > 0; allp["core_start"] = allp.s_first.fillna(allp.c_start.where(allp.c_status == "Live")); FLOW = 14
NEXT = pd.Timestamp("2026-11-23"); allp["cast_week_now"] = allp.week.where(allp.week >= NEXT, NEXT).where(allp.week.notna()); allp["castings_ready"] = allp.cast_week_now + pd.Timedelta(days=FLOW); allp["gap_cast_to_core_days"] = (allp.core_start - allp.castings_ready).dt.days; allp["gap_ready_to_required_days"] = (allp.required - allp.castings_ready).dt.days; allp["core_to_required_days"] = (allp.required - allp.core_start).dt.days
def risk(r):
    if not r.open: return None
    if pd.isna(r.week): return "No week on the PCT plan"
    if pd.notna(r.required) and r.cast_week_now + pd.Timedelta(days=4) > r.required: return "Cannot be cast before the required date"
    if pd.notna(r.core_start) and r.core_start < r.castings_ready: return "Core scheduled before castings are ready"
    if pd.notna(r.required) and pd.notna(r.core_wo) and pd.isna(r.core_start): return "Needs machining; no Core date"
    if pd.notna(r.required) and r.gap_ready_to_required_days < 7 and pd.notna(r.core_wo): return "Under a week between castings ready and required date"
    if r.week < WK: return "Past its plan week, still outstanding"
    return "No conflict found"
allp["risk"] = allp.apply(risk, axis=1); op = allp[allp.open]
save("p_overlap", op.sort_values(["week", "line"])[["line", "week", "customer", "works_order", "part_no", "qty_order", "qty_made", "outstanding", "core_wo", "s_machine", "core_start", "c_status", "required", "value", "so", "castings_ready", "gap_cast_to_core_days", "gap_ready_to_required_days", "risk", "notes"]].rename(columns={"s_machine": "core_machine"}), f"Open PCT plan lines (quantity outstanding) joined to Core Engineering and the order book. Core works order = the casting works order number, or one below or above it (machining orders are raised as the neighbouring number). Castings ready = plan week (or the week of 23 Nov 2026 if the plan week has passed with castings still outstanding) + {FLOW} days (2026 median mould-to-despatch is 18 days; fettling follows moulding by about 13). Core start = first date the works order appears on Core's schedule, else its live Continuum task date.")
RO = ["Cannot be cast before the required date", "Core scheduled before castings are ready", "Needs machining; no Core date", "Under a week between castings ready and required date", "Past its plan week, still outstanding", "No week on the PCT plan", "No conflict found"]
save("p_risk", [{"risk": k, "orders": int((op.risk == k).sum()), "castings_outstanding": op[op.risk == k].outstanding.sum(), "order_value": op[op.risk == k].value.sum(), "customers": ", ".join(sorted(set(op[op.risk == k].customer.dropna().astype(str).str.title())))[:70]} for k in RO if (op.risk == k).any()], "Open plan lines by the first conflict found, in order of severity. Order value is the open order value of the linked order book lines, where a link exists.")
# order book lines that need castings and are on neither plan
plan_wos = set(allp.works_order) | set(allp.core_wo.dropna()); started = set(d.wo)
obp = ob[~ob.part_no.str.contains(r"CARRIAGE|C OF C|CHARGE|PATTERN", case=False)].copy()
obp["on_pct_plan"] = obp.wos.map(lambda w: any((x in plan_wos) or (str(int(x) + 1) in plan_wos) or (str(int(x) - 1) in plan_wos) for x in w)); obp["on_core"] = obp.wos.map(lambda w: any((x in core_wo.index) or (x in sch_wo.index) for x in w)); obp["started"] = obp.wos.map(lambda w: any(x in started for x in w))
obp["days_to_required"] = (obp.required - TODAY).dt.days; miss = obp[~obp.on_pct_plan & ~obp.on_core & ~obp.started]
save("p_book_cover", [{"open_lines": len(obp), "value": obp.os_net.sum(), "on_pct_plan_lines": int(obp.on_pct_plan.sum()), "on_pct_plan_value": obp[obp.on_pct_plan].os_net.sum(), "on_core_lines": int(obp.on_core.sum()), "on_core_value": obp[obp.on_core].os_net.sum(), "on_both_lines": int((obp.on_pct_plan & obp.on_core).sum()), "on_both_value": obp[obp.on_pct_plan & obp.on_core].os_net.sum(), "core_only_value": obp[~obp.on_pct_plan & obp.on_core].os_net.sum(), "pct_only_value": obp[obp.on_pct_plan & ~obp.on_core].os_net.sum(), "on_neither_lines": int((~obp.on_pct_plan & ~obp.on_core).sum()), "on_neither_value": obp[~obp.on_pct_plan & ~obp.on_core].os_net.sum(), "on_neither_not_started_lines": len(miss), "on_neither_not_started_value": miss.os_net.sum(), "on_neither_due_60_days_value": miss[miss.days_to_required <= 60].os_net.sum(),
     "plan_open_orders": len(op), "plan_open_linked_to_core": int(op.core_wo.notna().sum()), "plan_open_linked_to_order_book": int(op.ob_wo.notna().sum()), "plan_open_castings": op.outstanding.sum(), "plan_rows": len(allp), "actual_castings_per_week_2026": act_all, "actual_castings_per_week_since_aug": act_recent}], "How much of the open order book appears on the PCT plan, on Core's plan (schedule or Continuum), on both or on neither. Pattern and charge lines left out.")
op2 = op.copy(); zx = op2[op2.risk == "Core scheduled before castings are ready"]
save("p_starve", [{"core_machine": m, "orders": len(g), "castings_outstanding": g.outstanding.sum(), "first_core_date": g.core_start.min(), "castings_ready_from": g.castings_ready.min(), "order_value": g.value.sum(), "parts": ", ".join(sorted(set(g.part_no.astype(str))))[:60]} for m, g in zx.groupby(zx.s_machine.fillna("not on schedule"))], "Core machines whose scheduled work depends on castings the PCT plan has not yet made.")
save("p_book_missing", [{"customer": r.customer, "part_no": r.part_no, "so": r.so, "required": r.required, "days_to_required": r.days_to_required, "units": r.os_qty, "value": r.os_net, "works_orders": r.works_orders} for r in miss.sort_values("os_net", ascending=False).head(15).itertuples()], "Largest open order lines on neither plan and with no shop-floor booking.")
print("lm", lm.to_dict()); print(allp[["core_wo", "ob_wo"]].notna().mean().round(2).to_dict(), (allp.core_wo == allp.works_order).mean(), (allp.core_wo.astype(float) == allp.n - 1).mean(), (allp.core_wo.astype(float) == allp.n + 1).mean())
print("report7 done", flush=True)
