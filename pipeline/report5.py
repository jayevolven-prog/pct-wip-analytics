"""PCT: Core Engineering machining (Continuum) and the open order book traced to works orders.

Usage: python report5.py <wip2.parquet> <report_dir> <ext_dir>
Reads PCT_Core_Works_Orders_Continuum.xlsx and PCT_Order_Book_*.xlsx. Writes JSON with prefix c_.
"""
import sys, os, json, math, re, glob, warnings
import numpy as np, pandas as pd
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
    json.dump({"rows": rows, "meta": [{"label": "report5.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]}, open(f"{OUT}/c_{name}.json", "w"), allow_nan=False); print("wrote c_" + name, len(rows), flush=True)
norm = lambda s: re.sub(r"\s+", "", str(s).upper())
TODAY = pd.Timestamp("2026-11-20")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["wo"] = d.wo_base.astype(str); d["key"] = d.part_no.map(norm)
c = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Core_Works_Orders_Continuum*.xlsx"))[-1], sheet_name="Tasks", dtype={"works_order": str})
c["wos"] = c.works_order.fillna("").str.findall(r"\d{5}"); c["hrs"] = (c.finish - c.start).dt.total_seconds() / 3600; c["key"] = c.part_no.map(norm)
fin = c[c.status == "Finished"].copy(); fin["month"] = fin.finish.fillna(fin.start).dt.to_period("M")
live = c[c.status.isin(["Scheduled", "Started", "Unscheduled"])]

# ---- Core summary
old26 = d[(d.op.between(100, 199)) & (d.date >= "2026-01-01")]; old25 = d[(d.op.between(100, 199)) & (d.date.dt.year == 2025)]
f26 = fin[fin.month >= "2026-01"]; m26 = TODAY.dayofyear / 30.4375
save("core_summary", [{"tasks": len(c), "first_task": c.start.min(), "finished": len(fin), "live": len(live), "cancelled": int(c.status.isin(["Cancelled", "Aborted"]).sum()), "parts": c.part_no.nunique(), "machines": int(c[c.machine != "Unassigned"].machine.nunique()), "units_finished": fin.current_qty.sum(), "units_expected_on_finished": fin.expected_qty.sum(),
    "tasks_with_pct_wo_pct": 100 * (c.wos.str.len() > 0).mean(), "parts_matched_to_pct_pct": 100 * c.drop_duplicates("key").key.isin(set(d.key)).mean(), "finished_2026": len(f26), "units_finished_2026": f26.current_qty.sum(), "units_per_month_2026": f26.current_qty.sum() / m26, "tasks_per_month_2026": len(f26) / m26,
    "finished_zero_qty_pct": 100 * (fin.current_qty == 0).mean(), "finished_short_pct": 100 * (fin.current_qty < fin.expected_qty).mean(), "finished_under_6min_pct": 100 * (fin.hrs < 0.1).mean(), "finished_no_finish_time_pct": 100 * fin.finish.isna().mean(), "unassigned_machine_pct": 100 * (c.machine == "Unassigned").mean(),
    "old_system_machining_units_2025": old25.qty.sum(), "old_system_machining_units_2026": old26.qty.sum(), "old_system_machining_entries_2026": len(old26), "old_system_timed_2026_pct": 100 * (old26.hrs_clean.fillna(0) > 0).mean() if len(old26) else None,
    "live_units": live.expected_qty.sum(), "live_overdue": int((live.start < TODAY).sum()), "live_unscheduled": int((live.status == "Unscheduled").sum())}],
    "Core Engineering (Cullompton) works orders from Continuum, 16 Jun 2025 to 20 Nov 2026 plus tasks scheduled to Jan 2027. Old system = machining operations (100 to 199) booked in the PCT shop-floor system.")
# monthly handover
om = d[d.op.between(100, 199) & (d.date >= "2025-01-01")].groupby(d.date.dt.to_period("M")).agg(units=("qty", "sum"), entries=("qty", "size"), timed=("hrs_clean", lambda s: (s.fillna(0) > 0).sum()), hrs=("hrs_clean", "sum"))
cm = fin.groupby("month").agg(units=("current_qty", "sum"), tasks=("task_id", "size")); md = d[(d.work_centre == "MDESP") & (d.date >= "2025-01-01")].groupby(d.date.dt.to_period("M")).qty.sum()
idx = pd.period_range("2025-01", "2026-11", freq="M")
save("core_handover", [{"month": str(m), "pct_machining_units": float(om.units.get(m, 0)), "pct_machining_entries": int(om.entries.get(m, 0)), "pct_timed_entries": int(om.timed.get(m, 0)), "pct_machining_hrs": float(om.hrs.get(m, 0)), "core_units_finished": float(cm.units.get(m, 0)), "core_tasks_finished": int(cm.tasks.get(m, 0)), "machined_despatch_units": float(md.get(m, 0))} for m in idx],
     "Machining by month in the two systems: units and timed entries booked at PCT machining operations, units finished on Core Engineering tasks in Continuum, and units booked through PCT machined despatch. November 2026 is a part month.")
# machines
mc = []
for m, g in c.groupby("machine"):
    f = g[g.status == "Finished"]; lv = g[g.status.isin(["Scheduled", "Started", "Unscheduled"])]; f6 = f[f.finish.fillna(f.start) >= "2026-01-01"]
    mc.append({"machine": m, "tasks": len(g), "finished": len(f), "units_finished": f.current_qty.sum(), "units_expected": f.expected_qty.sum(), "completion_pct": 100 * f.current_qty.sum() / f.expected_qty.sum() if f.expected_qty.sum() else None, "parts": g.part_no.nunique(), "units_per_month_2026": f6.current_qty.sum() / m26, "median_batch": f.expected_qty.median(), "queued_tasks": len(lv), "queued_units": lv.expected_qty.sum(), "weeks_of_work_queued": lv.expected_qty.sum() / (f6.current_qty.sum() / m26 / 4.35) if f6.current_qty.sum() else None, "first": g.start.min(), "last_scheduled": lv.start.max()})
save("core_machine", sorted(mc, key=lambda r: -r["units_finished"]), "Core Engineering by machine: tasks, units finished against expected, 2026 run rate and work queued (scheduled, started or unscheduled).")
# parts
pp = fin.groupby("part_no").agg(description=("description", "first"), tasks=("task_id", "size"), units=("current_qty", "sum"), expected=("expected_qty", "sum"), machines=("machine", lambda s: ", ".join(sorted(set(s) - {"Unassigned"}))[:40]), first=("start", "min"), last=("finish", "max")).reset_index()
pp["completion_pct"] = 100 * pp.units / pp.expected.replace(0, np.nan); pp["units_per_month_2026"] = pp.part_no.map(f26.groupby("part_no").current_qty.sum()).fillna(0) / m26
save("core_parts", pp.sort_values("units", ascending=False).head(25), "Parts machined at Core Engineering, by units finished.")
# flow through Core for linked works orders
ex = c.explode("wos").dropna(subset=["wos"]); ex = ex[ex.wos != ""]
cw = ex.groupby("wos").agg(core_start=("start", "min"), core_finish=("finish", "max"), core_status=("status", lambda s: "Finished" if (s == "Finished").all() else "Started" if (s.isin(["Started", "Finished"])).any() else "Scheduled" if (s == "Scheduled").any() else s.iloc[0]), core_machine=("machine", "first"), core_qty=("current_qty", "sum"))
st = d.groupby("wo").agg(first=("date", "min"), key=("key", "first"), part=("part_no", "first"))
stg = lambda lo, hi, f: d[d.op.between(lo, hi)].groupby("wo").date.agg(f)
st["mould"] = stg(10, 19, "min"); st["fettle"] = stg(20, 59, "min"); st["cdesp"] = stg(80, 99, "max"); st["pmach"] = stg(100, 199, "min"); st["mdesp"] = stg(280, 299, "max"); st["last"] = d.groupby("wo").date.max()
lk = cw.join(st, how="left"); lk["core_days"] = (lk.core_finish - lk.core_start).dt.total_seconds() / 86400; lk["finish_to_mdesp"] = (lk.mdesp - lk.core_finish.dt.normalize()).dt.days; lk["pct_first_to_core_start"] = (lk.core_start.dt.normalize() - lk["first"]).dt.days
g = lk[lk.core_status == "Finished"]
save("core_flow", [{"works_orders_linked": len(lk), "found_in_pct_bookings": int(lk["first"].notna().sum()), "with_machined_despatch_booking": int(lk.mdesp.notna().sum()), "core_days_median": g.core_days[g.core_days > 0.05].median(), "core_days_p75": g.core_days[g.core_days > 0.05].quantile(.75), "orders_with_real_duration": int((g.core_days > 0.05).sum()), "finish_to_despatch_median": g.finish_to_mdesp[(g.finish_to_mdesp >= 0) & (g.finish_to_mdesp < 200)].median(), "finish_to_despatch_p75": g.finish_to_mdesp[(g.finish_to_mdesp >= 0) & (g.finish_to_mdesp < 200)].quantile(.75), "despatch_booked_before_core_finish_pct": 100 * (g.finish_to_mdesp < 0).sum() / max(g.finish_to_mdesp.notna().sum(), 1), "pct_first_booking_to_core_start_median": lk.pct_first_to_core_start[lk.pct_first_to_core_start >= 0].median()}],
     "PCT works orders named in Continuum task titles, joined to PCT bookings on the same works order number. Core days = first task start to last task finish, only where a real duration was recorded.")

# ---- open order book traced to works orders
ob = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Order_Book_*.xlsx"))[-1], dtype={"works_orders": str, "part_no": str, "so": str})
ob["wos"] = ob.works_orders.fillna("").str.findall(r"\d{5}"); ob["charge"] = ob.part_no.str.contains(r"CARRIAGE|C OF C|SURCHARGE|CHARGE", case=False)
ob["days_to_required"] = (pd.to_datetime(ob.required) - TODAY).dt.days; ob["key"] = ob.part_no.map(norm)
core_parts = set(c.key); mach_keys = set(d[d.op >= 100].key)
ORDER = ["No works order raised", "Works order raised, not started", "In the foundry", "Cast, waiting for machining", "At Core: scheduled", "At Core: in progress", "Machined, waiting for despatch", "Part despatched"]
def stage(r):
    w = r.wos
    if not w: return "No works order raised"
    s = st.reindex(w); cs = cw.reindex(w); needs_m = r.key in core_parts or r.key in mach_keys or cs.core_status.notna().any()
    if s.mdesp.notna().any() or (r.shipped > 0 and not needs_m): return "Part despatched"
    if (cs.core_status == "Finished").any() and not cs.core_status.isin(["Started", "Scheduled", "Unscheduled"]).any(): return "Machined, waiting for despatch"
    if (cs.core_status == "Started").any() or (cs.core_status == "Finished").any(): return "At Core: in progress"
    if cs.core_status.isin(["Scheduled", "Unscheduled"]).any(): return "At Core: scheduled"
    if s.cdesp.notna().any() or (s.fettle.notna().any() and needs_m): return "Cast, waiting for machining" if needs_m else "Part despatched"
    if s["first"].notna().any(): return "In the foundry"
    return "Works order raised, not started"
ob["stage"] = ob.apply(stage, axis=1); prod = ob[~ob.charge]
def agg(g): return {"lines": len(g), "units": g.os_qty.sum(), "value": g.os_net.sum(), "overdue_value": g[g.days_to_required < 0].os_net.sum(), "due_30_days_value": g[(g.days_to_required >= 0) & (g.days_to_required <= 30)].os_net.sum(), "due_31_90_value": g[(g.days_to_required > 30) & (g.days_to_required <= 90)].os_net.sum(), "later_value": g[g.days_to_required > 90].os_net.sum(), "median_days_to_required": g.days_to_required.median()}
save("book_stage", [{"stage": s, **agg(prod[prod.stage == s]), "share_pct": 100 * prod[prod.stage == s].os_net.sum() / prod.os_net.sum()} for s in ORDER if (prod.stage == s).any()] , "Open order lines at 20 Nov 2026 (charges such as carriage and certificates left out), placed at the furthest stage reached by the works orders raised against each line: PCT bookings for foundry stages, Continuum task status for machining at Core.")
risk = prod[(prod.days_to_required <= 30) & prod.stage.isin(ORDER[:5])].sort_values("os_net", ascending=False)
save("book_at_risk", [{"customer": r.customer, "part_no": r.part_no, "so": r.so, "required": r.required, "days_to_required": r.days_to_required, "units": r.os_qty, "value": r.os_net, "stage": r.stage, "works_orders": r.works_orders} for r in risk.head(25).itertuples()], "Open lines overdue or due within 30 days that have not yet started machining or are further back.")
save("book_summary", [{"lines": len(ob), "product_lines": len(prod), "value": ob.os_net.sum(), "product_value": prod.os_net.sum(), "customers": ob.customer.nunique(), "lines_with_wo_pct": 100 * (prod.wos.str.len() > 0).mean(), "value_with_wo_pct": 100 * prod[prod.wos.str.len() > 0].os_net.sum() / prod.os_net.sum(), "no_wo_value": prod[prod.stage == ORDER[0]].os_net.sum(), "no_wo_due_90_value": prod[(prod.stage == ORDER[0]) & (prod.days_to_required <= 90)].os_net.sum(), "not_started_value": prod[prod.stage.isin(ORDER[:2])].os_net.sum(), "needs_core_value": prod[prod.key.isin(core_parts) | prod.key.isin(mach_keys)].os_net.sum(), "needs_core_pct": 100 * prod[prod.key.isin(core_parts) | prod.key.isin(mach_keys)].os_net.sum() / prod.os_net.sum(),
    "at_risk_lines": len(risk), "at_risk_value": risk.os_net.sum(), "overdue_value": prod[prod.days_to_required < 0].os_net.sum(), "due_30_value": prod[(prod.days_to_required >= 0) & (prod.days_to_required <= 30)].os_net.sum(), "wo_linked": int(prod.wos.explode().dropna().nunique()), "wo_found_in_pct_pct": 100 * prod.wos.explode().dropna().drop_duplicates().isin(set(d.wo)).mean(), "wo_found_in_core_pct": 100 * prod.wos.explode().dropna().drop_duplicates().isin(set(cw.index)).mean()}], "Open order book file dated 20 Nov 2026 and how much of it can be traced to works orders.")
# true order-to-start on open lines (real links)
tl = prod[prod.wos.str.len() > 0].copy(); tl["first_booking"] = tl.wos.map(lambda w: st.reindex(w)["first"].min()); tl["to_start"] = (tl.first_booking - pd.to_datetime(tl.entered)).dt.days; tl["allowed"] = (pd.to_datetime(tl.required) - pd.to_datetime(tl.entered)).dt.days
t2 = tl[tl.to_start.notna()]
save("book_trace", [{"lines_traced": len(t2), "order_to_first_booking_median": t2.to_start.median(), "order_to_first_booking_p75": t2.to_start.quantile(.75), "allowed_median": t2.allowed.median(), "share_of_allowance_used_before_start_pct": 100 * (t2.to_start / t2.allowed.replace(0, np.nan)).median(), "lines_with_wo_not_started": int(tl.to_start.isna().sum()), "their_median_days_since_order": (TODAY - pd.to_datetime(tl[tl.to_start.isna()].entered)).dt.days.median(), "their_median_days_to_required": tl[tl.to_start.isna()].days_to_required.median()}], "Open order lines with a works order: days from order entry to the first shop-floor booking on that works order. A true trace, unlike the part-and-date link used for closed orders.")
# fettled castings with no PCT despatch booking: how many are machined-route work that leaves via Core
rec = st[(st.fettle > pd.Timestamp("2026-11-19") - pd.Timedelta(days=120)) & st.cdesp.isna()].copy(); mq = d[d.op.between(10, 19)].groupby("wo").qty.sum(); rec["qty"] = mq.reindex(rec.index)
rec = rec[rec.qty > 0]; rec["core_task"] = rec.index.isin(cw.index); rec["machined_part"] = rec.key.isin(core_parts) | rec.key.isin(mach_keys) | rec.mdesp.notna() | rec.pmach.notna()
save("stock_at_core", [{"orders": len(rec), "castings": rec.qty.sum(), "orders_with_core_task": int(rec.core_task.sum()), "castings_with_core_task": rec[rec.core_task].qty.sum(), "orders_machined_route": int((rec.core_task | rec.machined_part).sum()), "castings_machined_route": rec[rec.core_task | rec.machined_part].qty.sum(), "castings_cast_only_unbooked": rec[~(rec.core_task | rec.machined_part)].qty.sum(), "machined_route_share_pct": 100 * rec[rec.core_task | rec.machined_part].qty.sum() / rec.qty.sum()}], "Works orders fettled in the last 120 days with no PCT casting-despatch booking, split by whether the order has a Core Engineering task or a machining route.")
# queue at Core by machine and month
lv = live.copy(); lv["month"] = lv.start.dt.to_period("M").astype(str).replace("NaT", "Unscheduled")
save("core_queue", [{"month": m, "tasks": len(g), "units": g.expected_qty.sum(), "machines": g.machine.nunique()} for m, g in lv.groupby("month")], "Live Core Engineering tasks (scheduled, started, unscheduled) by scheduled start month. Tasks dated before 20 Nov 2026 are behind schedule.")
print("report5 done", flush=True)
