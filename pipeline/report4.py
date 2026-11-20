"""PCT sales order and scrap analytics, linked to the shop-floor bookings.

Usage: python report4.py <wip2.parquet> <report_dir> <ext_dir>
Reads PCT_Sales_Orders_*.xlsx and PCT_Scrap_Analysis_*.xlsx from ext_dir. Writes JSON with prefix s_.
"""
import sys, os, json, math, re, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SRC, OUT, EXT = sys.argv[1:4]
CODE = open(os.path.abspath(__file__)).read()
def clean(v):
    if isinstance(v, (np.floating, float)): return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, pd.Timestamp): return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    if v is pd.NaT: return None
    if isinstance(v, (np.bool_,)): return bool(v)
    return v
def save(name, rows, note=""):
    if isinstance(rows, pd.DataFrame): rows = rows.to_dict("records")
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [{"label": "report4.py", "value": CODE, "type": "code", "lang": "python"}, {"label": "Note", "value": note}]
    json.dump({"rows": rows, "meta": meta}, open(f"{OUT}/s_{name}.json", "w"), allow_nan=False); print("wrote s_" + name, len(rows), flush=True)
J = lambda n: json.load(open(f"{OUT}/{n}.json"))["rows"]
norm = lambda s: re.sub(r"\s+", "", str(s).upper())

so = pd.read_excel(sorted(glob.glob(f"{EXT}/PCT_Sales_Orders_*.xlsx"))[-1], sheet_name="Sales order lines", dtype={"so_no": str, "so": str, "order_no": str, "part_no": str, "contract_no": str})
for c in ("entered", "required", "promised"): so[c] = pd.to_datetime(so[c]).astype("datetime64[ns]")
END = so.entered.max(); ann = 365 / END.dayofyear; so["yr"] = so.entered.dt.year; so["key"] = so.part_no.map(norm)
NONPROD = r"CARRIAGE|C OF C|SURCHARGE|CHARGE|PATTERN|TOOLING|MODIFICATION|CORE ?BOX|FIXTURE|GAUGE|^TOOL|NRE|SAMPLE"
so["kind"] = np.where(so.part_no.str.contains(NONPROD, case=False, regex=True) | so.description.fillna("").str.contains(r"^(PATTERN|TOOLING|CARRIAGE|CORE ?BOX)", case=False, regex=True), "Charges and tooling", "Product")
d = pd.read_parquet(SRC); d["date"] = d.date.astype("datetime64[ns]"); d["op"] = pd.to_numeric(d["wo_op"], errors="coerce"); d["key"] = d.part_no.map(norm); WEND = d.date.max()
_ck = set(d[d.op.between(10, 19)].key)
def ckey(k): return k if k in _ck else (k + "C" if k + "C" in _ck else (k + "-C" if k + "-C" in _ck else None))
so["ckey"] = so.key.map(ckey); so["stream"] = np.where(so.ckey.notna(), "Sold under a casting part number", np.where(so.key.isin(set(d[d.op >= 100].key)), "Sold under a machined part number", "No shop-floor bookings"))
live = so[so.status != "Cancelled"]; pl = live[live.kind == "Product"]; canc = so[so.status == "Cancelled"]
_rev = pl.dropna(subset=["ckey"]).sort_values("entered").groupby("ckey").key.last()
YRS = [2022, 2023, 2024, 2025, 2026]; A = lambda y: ann if y == 2026 else 1.0
price_any = (pl[pl.ordered > 0].groupby("key").order_net.sum() / pl[pl.ordered > 0].groupby("key").ordered.sum())
last_price = pl[(pl.ordered > 0) & (pl.order_net > 0)].sort_values("entered").groupby("key").apply(lambda g: g.order_net.iloc[-3:].sum() / g.ordered.iloc[-3:].sum())

# ===== A. sales by year
ys = []
for y in YRS:
    g, p, c = live[live.yr == y], pl[pl.yr == y], canc[canc.yr == y]
    cv = (c.ordered * c.key.map(last_price)).sum()
    ys.append({"year": str(y), "order_lines": len(g), "customers": g.customer.nunique(), "parts": p.part_no.nunique(), "units": p.ordered.sum(), "value": g.order_net.sum(), "product_value": p.order_net.sum(), "charges_tooling_value": g.order_net.sum() - p.order_net.sum(), "cast_part_value": p[p.ckey.notna()].order_net.sum(), "machined_part_value": p[p.ckey.isna()].order_net.sum(), "cast_part_units": p[p.ckey.notna()].ordered.sum(), "machined_part_units": p[p.ckey.isna()].ordered.sum(),
               "value_annualised": g.order_net.sum() * A(y), "units_annualised": p.ordered.sum() * A(y), "avg_price_per_unit": p.order_net.sum() / p.ordered.sum(), "median_line_value": p.order_net.median(), "cancelled_lines": len(c), "cancelled_units": c.ordered.sum(), "cancelled_value_est": cv, "cancelled_share_of_lines_pct": 100 * len(c) / (len(c) + len(g))})
save("sales_year", ys, f"Sales order lines by year entered (file covers 22 Nov 2021 to {END:%d %b %Y}; 2021 omitted as a part year). Cancelled lines carry no value in the export, so cancelled value is estimated at the part's recent selling price. 2026 annualised by {ann:.3f}.")
# price index (chained Laspeyres on matched parts)
py = pl[(pl.ordered > 0) & (pl.order_net > 0)].groupby(["key", "yr"]).agg(q=("ordered", "sum"), v=("order_net", "sum")); py["p"] = py.v / py.q; pw = py.p.unstack(); qw = py.q.unstack()
idx = {2022: 100.0}; cov = {2022: None}
for y in YRS[1:]:
    m = pw[y].notna() & pw[y - 1].notna(); idx[y] = idx[y - 1] * (pw[y][m] * qw[y - 1][m]).sum() / (pw[y - 1][m] * qw[y - 1][m]).sum(); cov[y] = 100 * (pw[y][m] * qw[y][m]).sum() / (pw[y] * qw[y]).sum()
cy = {int(r["year"]): r for r in J("x_commercial_year")}
save("price_index", [{"year": str(y), "price_index": idx[y], "matched_share_of_value_pct": cov[y], "wage_floor_index": 100 * cy[y]["loaded_min_rate"] / cy[2022]["loaded_min_rate"], "aluminium_index": 100 * cy[y]["lme_usd_t"] / cy[2022]["lme_usd_t"], "avg_price_per_unit": pl[pl.yr == y].order_net.sum() / pl[pl.yr == y].ordered.sum()} for y in YRS], "Chained Laspeyres selling-price index on parts sold in consecutive years (previous-year quantities as weights), 2022 = 100, beside the minimum loaded wage rate and LME aluminium. Average price per unit also moves with mix.")

# ===== B. customers
cv_ = live.groupby(["customer", "yr"]).order_net.sum().unstack(fill_value=0).reindex(columns=YRS, fill_value=0); cu = pl.groupby(["customer", "yr"]).ordered.sum().unstack(fill_value=0).reindex(columns=YRS, fill_value=0)
cv_["v26a"] = cv_[2026] * ann; cv_ = cv_.sort_values(2026, ascending=False); t26 = cv_[2026].sum()
osn = so[so.os_net > 0].groupby("customer").os_net.sum()
cust = [{"customer": c, **{f"value_{y}": r[y] for y in YRS}, "value_2026_annualised": r["v26a"], "change_vs_2025_pct": 100 * (r["v26a"] / r[2025] - 1) if r[2025] > 0 else None, "share_2026_pct": 100 * r[2026] / t26, "units_2025": cu.loc[c, 2025] if c in cu.index else 0, "units_2026_annualised": (cu.loc[c, 2026] * ann) if c in cu.index else 0, "open_order_value": osn.get(c, 0.0), "parts_2026": pl[(pl.customer == c) & (pl.yr == 2026)].part_no.nunique(), "avg_price_2025": (pl[(pl.customer == c) & (pl.yr == 2025)].order_net.sum() / max(pl[(pl.customer == c) & (pl.yr == 2025)].ordered.sum(), 1)), "avg_price_2026": (pl[(pl.customer == c) & (pl.yr == 2026)].order_net.sum() / max(pl[(pl.customer == c) & (pl.yr == 2026)].ordered.sum(), 1))} for c, r in cv_.iterrows()]
save("customers", cust[:25], "Order value by customer and year entered, largest 2026 customers first.")
sh = (cv_[YRS].div(cv_[YRS].sum()) * 100)
save("customer_concentration", [{"year": str(y), "customers": int((cv_[y] > 0).sum()), "top1_share_pct": sh[y].nlargest(1).sum(), "top3_share_pct": sh[y].nlargest(3).sum(), "top5_share_pct": sh[y].nlargest(5).sum(), "hhi": (sh[y] ** 2).sum(), "top_customer": sh[y].idxmax()} for y in YRS], "Customer concentration by order value. HHI above 2,500 is highly concentrated.")
mv = (cu[2026] * ann - cu[2023]).sort_values()
save("customer_unit_moves", [{"customer": c, "units_2023": cu.loc[c, 2023], "units_2026_annualised": cu.loc[c, 2026] * ann, "change_units": v, "value_2023": cv_.loc[c, 2023], "value_2026_annualised": cv_.loc[c, "v26a"]} for c, v in list(mv.head(8).items()) + list(mv.tail(5)[::-1].items())], "Customers with the largest change in units ordered, 2023 to 2026 (annualised). First eight fell, last five rose.")

# ===== C. order book
ob = so[(so.os_net > 0) & (so.status != "Cancelled") & (so.status != "Invoiced")].copy(); ob["days_to_required"] = (ob.required - END).dt.days
ob["bucket"] = pd.cut(ob.days_to_required, [-9999, -31, -1, 30, 90, 180, 9999], labels=["Overdue over 30 days", "Overdue up to 30 days", "Due in 30 days", "Due in 31 to 90 days", "Due in 91 to 180 days", "Due beyond 180 days"])
save("order_book", [{"bucket": str(b), "lines": len(g), "units_outstanding": g.os_qty.sum(), "value_outstanding": g.os_net.sum(), "customers": g.customer.nunique(), "share_pct": 100 * g.os_net.sum() / ob.os_net.sum()} for b, g in ob.groupby("bucket")], f"Open order lines (not invoiced or cancelled) with outstanding value at {END:%d %b %Y}, by required date.")
save("order_book_customer", [{"customer": c, "lines": len(g), "value_outstanding": g.os_net.sum(), "overdue_value": g[g.days_to_required < 0].os_net.sum(), "due_90_days_value": g[(g.days_to_required >= 0) & (g.days_to_required <= 90)].os_net.sum(), "latest_required": g.required.max()} for c, g in sorted(ob.groupby("customer"), key=lambda x: -x[1].os_net.sum())][:15], "Open order book by customer.")
od = ob[ob.days_to_required < 0].sort_values("os_net", ascending=False)
save("order_book_overdue", [{"customer": r.customer, "part_no": r.part_no, "so_no": r.so_no, "status": r.status, "required": r.required, "days_overdue": -r.days_to_required, "units_outstanding": r.os_qty, "value_outstanding": r.os_net} for r in od.head(30).itertuples()], "Largest open lines past their required date.")
inv_os = so[(so.status == "Invoiced") & (so.os_net > 0)]
bm = ob.groupby(ob.required.dt.to_period("M")).os_net.sum()
save("order_book_month", [{"month": str(k), "value_outstanding": v} for k, v in bm.items()], "Open order value by month required.")

# ===== D. lead times
pl2 = pl[pl.required.notna()].copy(); pl2["asked"] = (pl2.required - pl2.entered).dt.days; pl2["prom_vs_req"] = (pl2.promised - pl2.required).dt.days
lt = []
for y in YRS:
    g = pl2[(pl2.yr == y) & (pl2.asked >= 0)]
    lt.append({"year": str(y), "lines": len(g), "asked_median_days": g.asked.median(), "asked_p25": g.asked.quantile(.25), "asked_p75": g.asked.quantile(.75), "under_4_weeks_pct": 100 * (g.asked < 28).mean(), "over_12_weeks_pct": 100 * (g.asked > 84).mean(), "promised_later_than_required_pct": 100 * (g.prom_vs_req > 0).mean(), "promised_earlier_pct": 100 * (g.prom_vs_req < 0).mean()})
save("lead_asked_year", lt, "Days between order entry and the customer's required date, product lines by year entered; and how often the promised date differs from the required date.")
# shop-floor: entry to first moulding, moulding to despatch
mo = d[d.op.between(10, 19)]; fm = mo.groupby(["key", "wo_base"]).date.min().reset_index().sort_values("date")
fdsp = d[d.op.between(80, 99)].groupby("wo_base").date.min(); fm["desp"] = fm.wo_base.map(fdsp); fm["flow"] = (fm.desp - fm.date).dt.days; fm.loc[(fm.flow < 0) | (fm.flow > 365), "flow"] = np.nan
cast_keys = set(fm.key); x = pl2[pl2.ckey.notna() & (pl2.entered >= "2022-01-01")].sort_values("entered")
mrg = pd.merge_asof(x, fm.rename(columns={"date": "first_mould"})[["key", "first_mould", "flow"]].sort_values("first_mould"), left_on="entered", right_on="first_mould", left_by="ckey", right_by="key", direction="forward", tolerance=pd.Timedelta(days=365))
mrg["to_start"] = (mrg.first_mould - mrg.entered).dt.days
ls = []
for y in YRS:
    g = mrg[mrg.yr == y]; gg = g[g.to_start.notna()]
    ls.append({"year": str(y), "cast_lines": len(g), "linked_pct": 100 * len(gg) / max(len(g), 1), "asked_median_days": g.asked.median(), "entry_to_first_mould_median": gg.to_start.median(), "entry_to_first_mould_p75": gg.to_start.quantile(.75), "mould_to_despatch_median": gg.flow[(gg.flow >= 0) & (gg.flow <= 365)].median(), "share_of_asked_time_before_moulding_pct": 100 * gg.to_start.median() / g.asked.median() if g.asked.median() else None})
save("lead_stages_year", ls, "Cast parts only. Entry to first mould: days from sales order entry to the next moulding booking of that part (within a year). The bookings carry no sales order number, so this is a link by part and date, not a true order trace.")
lc = []
for c, g in pl2[pl2.yr == 2026].groupby("customer"):
    if len(g) < 8: continue
    g5 = pl2[(pl2.yr == 2025) & (pl2.customer == c)]; m = mrg[(mrg.yr == 2026) & (mrg.customer == c)]
    lc.append({"customer": c, "lines_2026": len(g), "value_2026": g.order_net.sum(), "asked_median_2026": g.asked.median(), "asked_median_2025": g5.asked.median() if len(g5) else None, "entry_to_first_mould_median": m.to_start.median() if m.to_start.notna().sum() >= 5 else None, "promised_later_pct": 100 * (g.prom_vs_req > 0).mean()})
save("lead_customer_2026", sorted(lc, key=lambda r: -r["value_2026"])[:15], "Lead time asked by customer, orders entered in 2026.")
# estimated delivery by first-in first-out match of despatch bookings to order lines
dd = d[d.work_centre.isin(["DESP", "MDESP"]) & (d.date >= so.entered.min())].groupby(["key", "date"]).qty.sum().reset_index()
est = []; recon = []
for k, g in pl[(pl.shipped > 0) & pl.required.notna()].groupby("key"):
    s = dd[dd.key == k].sort_values("date")
    if s.empty: continue
    ratio = s.qty.sum() / g.shipped.sum(); recon.append(ratio)
    if not (0.7 <= ratio <= 1.4): continue
    g = g.sort_values(["required", "entered"]); cum = g.shipped.cumsum().values * min(ratio, 1.0); sc_ = s.qty.cumsum().values; pos = np.searchsorted(sc_, cum - 1e-9)
    g = g.assign(est_ship=[s.date.values[i] if i < len(s) else np.datetime64("NaT") for i in pos]); est.append(g)
est = pd.concat(est); est["late"] = (est.est_ship - est.required).dt.days; est = est[est.est_ship.notna() & (est.est_ship >= est.entered - pd.Timedelta(days=3))]
tot_lines = len(pl[(pl.shipped > 0) & pl.required.notna()])
ot = []
for lab, m in [(str(y), est.required.dt.year == y) for y in YRS[:-1]] + [("2026 Jan to Jul", (est.required >= "2026-01-01") & (est.required < "2026-08-01"))]:
    g = est[m]
    if len(g) < 30: continue
    ot.append({"required_in": lab, "lines_estimated": len(g), "on_time_pct": 100 * (g.late <= 0).mean(), "within_7_days_pct": 100 * (g.late <= 7).mean(), "over_30_days_late_pct": 100 * (g.late > 30).mean(), "median_days_late": g.late.median(), "late_value": g[g.late > 7].order_net.sum(), "value": g.order_net.sum()})
save("on_time_est", ot, f"Estimated delivery against required date. Despatch bookings for each part are matched to its shipped order lines first-in first-out by required date; parts are included only where booked despatches are 70 to 140% of shipped quantity ({len(est)} of {tot_lines} shipped lines). An estimate: the export has no ship date. Lines required from August 2026 are left out because despatch bookings are incomplete.")
e26 = est[(est.required >= "2026-01-01") & (est.required < "2026-08-01")]
save("on_time_customer_2026", [{"customer": c, "lines": len(g), "on_time_pct": 100 * (g.late <= 0).mean(), "within_7_days_pct": 100 * (g.late <= 7).mean(), "median_days_late": g.late.median(), "late_value": g[g.late > 7].order_net.sum()} for c, g in sorted(e26.groupby("customer"), key=lambda x: -len(x[1])) if len(g) >= 10], "Estimated delivery performance by customer, lines required January to July 2026.")

# ===== E. run rates per product
M26 = END.dayofyear / 30.4375
wm = mo.groupby(["key", mo.date.dt.year]).qty.sum().unstack(fill_value=0); wb = mo.groupby(["key", "wo_base"]).qty.sum().groupby("key").median()
fl26 = fm[fm.date.dt.year == 2026].groupby("key").flow.median(); flall = fm.groupby("key").flow.median()
rr = []
for k, g in pl.groupby("key"):
    ck_ = g.ckey.iloc[0]
    g6, g5 = g[g.yr == 2026], g[g.yr == 2025]
    if g6.ordered.sum() == 0 and g5.ordered.sum() == 0: continue
    a6 = (g6.required - g6.entered).dt.days
    rr.append({"part_no": g.part_no.iloc[-1], "description": g.description.dropna().iloc[-1] if g.description.notna().any() else None, "customer": g.customer.iloc[-1], "value_2026": g6.order_net.sum(), "value_2025": g5.order_net.sum(), "units_per_month_2026": g6.ordered.sum() / M26, "units_per_month_2025": g5.ordered.sum() / 12, "units_per_month_2022_2024": g[g.yr.between(2022, 2024)].ordered.sum() / 36,
               "run_rate_change_pct": 100 * ((g6.ordered.sum() / M26) / (g5.ordered.sum() / 12) - 1) if g5.ordered.sum() > 0 else None, "orders_2026": g6.so.nunique(), "median_order_qty": g6.ordered.median() if len(g6) else g5.ordered.median(), "stream": g.stream.iloc[0].replace("Sold under a ", "").replace(" part number", ""), "median_batch_moulded": wb.get(ck_), "moulded_per_month_2026": (wm.loc[ck_, 2026] / (WEND.dayofyear / 30.4375)) if (ck_ in wm.index and 2026 in wm) else None,
               "price_2026": g6.order_net.sum() / g6.ordered.sum() if g6.ordered.sum() > 0 else None, "price_2025": g5.order_net.sum() / g5.ordered.sum() if g5.ordered.sum() > 0 else None, "asked_lead_median_2026": a6.median() if len(a6) else None, "shop_flow_median_2026": fl26.get(ck_), "shop_flow_median_all": flall.get(ck_), "units_outstanding": g.os_qty[g.status != "Invoiced"].sum(), "value_outstanding": g.os_net[g.status != "Invoiced"].sum()})
rr = pd.DataFrame(rr)
save("run_rates", rr.sort_values("value_2026", ascending=False).head(40), f"Run rate per product: units ordered per month by year entered (2026 over {M26:.1f} months), typical order and batch size, price, lead time asked and shop-floor flow time (first mould to first despatch booking). Top 40 by 2026 order value.")
both = rr[(rr.units_per_month_2025 > 0) & (rr.value_2025 + rr.value_2026 > 5000)].copy(); both["delta_value"] = both.value_2026 * ann - both.value_2025
save("run_rate_fallers", both.sort_values("delta_value").head(15)[["part_no", "customer", "units_per_month_2025", "units_per_month_2026", "run_rate_change_pct", "value_2025", "value_2026", "delta_value"]], "Products with the largest fall in annualised order value, 2025 to 2026.")
save("run_rate_risers", both.sort_values("delta_value", ascending=False).head(15)[["part_no", "customer", "units_per_month_2025", "units_per_month_2026", "run_rate_change_pct", "value_2025", "value_2026", "delta_value"]], "Products with the largest rise in annualised order value, 2025 to 2026.")
new26 = rr[(rr.value_2026 > 0) & (rr.units_per_month_2025 == 0) & (rr.units_per_month_2022_2024 == 0)]
save("run_rate_summary", [{"products_ordered_2026": int((rr.value_2026 > 0).sum()), "products_ordered_2025": int((rr.value_2025 > 0).sum()), "new_in_2026": len(new26), "new_value_2026": new26.value_2026.sum(), "stopped_since_2025": int(((rr.value_2025 > 0) & (rr.value_2026 == 0)).sum()), "stopped_value_2025": rr[(rr.value_2025 > 0) & (rr.value_2026 == 0)].value_2025.sum(), "top10_share_2026_pct": 100 * rr.value_2026.nlargest(10).sum() / rr.value_2026.sum(), "top20_share_2026_pct": 100 * rr.value_2026.nlargest(20).sum() / rr.value_2026.sum(), "median_order_qty_2026": pl[pl.yr == 2026].ordered.median(), "median_order_qty_2023": pl[pl.yr == 2023].ordered.median()}])
uy = [{"year": str(y), "units_ordered": pl[pl.yr == y].ordered.sum() * A(y), "castings_moulded": mo[mo.date.dt.year == y].qty.sum() * (365 / WEND.dayofyear if y == 2026 else 1), "castings_despatched": d[(d.work_centre == "DESP") & (d.date.dt.year == y)].qty.sum() * (365 / WEND.dayofyear if y == 2026 else 1), "machined_despatched": d[(d.work_centre == "MDESP") & (d.date.dt.year == y)].qty.sum() * (365 / WEND.dayofyear if y == 2026 else 1), "units_shipped_on_orders": pl[pl.yr == y].shipped.sum()} for y in YRS]
save("demand_vs_output", uy, "Units ordered (by year entered) beside castings moulded and units booked through despatch (by year booked). 2026 annualised except shipped.")

# ===== F. price against labour cost
rp = {norm(r["part_no"]): r for r in J("z_reprice_all")}; rate = {y: cy[y]["loaded_min_rate"] for y in cy}
pc = []
for k in pw.index:
    r = rp.get(ckey(k))
    if r is None: continue
    yrs_ = [y for y in YRS if pw.loc[k].get(y) == pw.loc[k].get(y)]
    if len(yrs_) < 2 or yrs_[-1] < 2025: continue
    y0, y1 = yrs_[0], yrs_[-1]; p0, p1 = pw.loc[k, y0], pw.loc[k, y1]; h = r["hrs_per_casting"]; l0, l1 = h * rate[y0], h * rate[2026]
    u = (qw.loc[k].get(2026, 0) or 0) * ann if qw.loc[k].get(2026) == qw.loc[k].get(2026) else 0
    if u == 0: u = qw.loc[k].get(2025, 0) if qw.loc[k].get(2025) == qw.loc[k].get(2025) else 0
    pc.append({"part_no": r["part_no"], "customer": pl[pl.key == k].customer.iloc[-1], "line": r["line"], "first_year": y0, "last_year": y1, "price_first": p0, "price_last": p1, "price_change_pct": 100 * (p1 / p0 - 1), "labour_first": l0, "labour_now": l1, "labour_change_pct": 100 * (l1 / l0 - 1), "labour_share_of_price_pct": 100 * l1 / p1, "price_rise_per_unit": p1 - p0, "labour_rise_per_unit": l1 - l0, "annual_units": u, "price_rise_annual": (p1 - p0) * u, "labour_rise_annual": (l1 - l0) * u, "real_price_change_pct": 100 * ((p1 / p0) / (rate[2026] / rate[y0]) - 1), "value_annual": p1 * u})
pc = pd.DataFrame(pc)
if len(pc):
    behind = pc[pc.price_change_pct < pc.labour_change_pct * 0.5].copy(); behind["shortfall_if_price_matched_wages"] = (behind.price_first * (1 + behind.labour_change_pct / 100) - behind.price_last) * behind.annual_units
    save("price_vs_labour", pc.sort_values("value_annual", ascending=False).head(40), "Cast parts sold in two or more years and still active: selling price change from first year in the file to latest, beside the change in minimum foundry labour cost per casting. Labour is the statutory floor on foundry attendance hours only; metal, energy, machining and overhead are not included.")
    save("price_behind", behind.sort_values("shortfall_if_price_matched_wages", ascending=False).head(25)[["part_no", "customer", "first_year", "price_first", "price_last", "price_change_pct", "labour_change_pct", "labour_share_of_price_pct", "annual_units", "value_annual", "shortfall_if_price_matched_wages"]], "Parts whose price has risen by less than half the rise in the wage floor since first ordered. Shortfall = extra annual revenue if the price had tracked the wage floor.")
    save("price_summary", [{"parts": len(pc), "annual_value": pc.value_annual.sum(), "median_price_change_pct": pc.price_change_pct.median(), "value_weighted_price_change_pct": 100 * (pc.price_last * pc.annual_units).sum() / (pc.price_first * pc.annual_units).sum() - 100, "median_labour_change_pct": pc.labour_change_pct.median(), "parts_price_flat_or_down": int((pc.price_change_pct <= 1).sum()), "value_price_flat_or_down": pc[pc.price_change_pct <= 1].value_annual.sum(), "parts_behind": len(behind), "value_behind": behind.value_annual.sum(), "shortfall_if_matched_wages": behind.shortfall_if_price_matched_wages.sum(), "labour_share_of_price_median_pct": pc.labour_share_of_price_pct.median(), "price_rise_annual_total": pc.price_rise_annual.sum(), "labour_rise_annual_total": pc.labour_rise_annual.sum()}])

# ===== G. scrap
sx = sorted(glob.glob(f"{EXT}/PCT_Scrap_Analysis_*.xlsx"))[-1]; SH = pd.read_excel(sx, sheet_name=None)
YEARS_SC = (pd.Timestamp("2026-11-20") - pd.Timestamp("2021-10-09")).days / 365.25
for nm, key in (("Scrap Reason", "scrap_reason"), ("Work Centre Cell", "scrap_cell"), ("Work Centre", "scrap_work_centre"), ("Group Code", "scrap_group"), ("Work Centre Location", "scrap_location"), ("Employee", "scrap_employee")):
    save(key, SH[nm].head(20), f"Scrap Analysis report, scrap dated 9 Oct 2021 to 20 Nov 2026 ({nm}).")
sp_ = SH["Part No"].copy(); sp_["key"] = sp_.item.map(norm); sp_["price"] = sp_.key.map(last_price).fillna(sp_.key.map(_rev).map(last_price)); sp_["scrap_value"] = sp_.scrap_qty * sp_.price
cust_of = pl.sort_values("entered").groupby("key").customer.last(); sp_["customer"] = sp_.key.map(cust_of).fillna(sp_.key.map(_rev).map(cust_of))
# shop-floor loss per part since Oct 2021
q = d[d.op.notna()].groupby(["wo_base", "op"]).qty.sum().unstack()
def col(lo, hi):
    c = [x for x in q.columns if lo <= x <= hi]; return q[c].sum(axis=1, min_count=1)
m_, dsp = col(10, 19), col(80, 99); st = d.groupby("wo_base").date.min(); kk = d.groupby("wo_base").key.first()
oo = pd.DataFrame({"m": m_, "g": dsp, "start": st, "key": kk}).dropna(subset=["m", "g"]); oo = oo[(oo.m > 0) & (oo.start >= "2021-10-09") & (oo.start < WEND - pd.Timedelta(days=45))]; oo["lost"] = oo.m - oo.g.clip(upper=oo.m)
wl = oo.groupby("key").agg(moulded=("m", "sum"), bookings_lost=("lost", "sum")); sp_ = sp_.join(wl, on="key")
sp_["gap"] = sp_.bookings_lost - sp_.scrap_qty
save("scrap_parts", sp_.sort_values("scrap_qty", ascending=False).head(25)[["item", "customer", "good_booked", "scrap_qty", "scrap_pct_of_booked", "price", "scrap_value", "moulded", "bookings_lost"]].rename(columns={"item": "part_no"}), "Parts with the most booked scrap, valued at recent selling price, beside castings the shop-floor bookings show as moulded but not despatched (orders started since 9 Oct 2021).")
allp = wl.join(sp_.set_index("key")[["scrap_qty", "item"]], how="outer"); allp["scrap_qty"] = allp.scrap_qty.fillna(0); allp["bookings_lost"] = allp.bookings_lost.fillna(0); allp["gap"] = allp.bookings_lost - allp.scrap_qty
nm_ = d.groupby("key").part_no.first(); allp["part_no"] = allp.item.fillna(pd.Series(allp.index.map(nm_), index=allp.index)); allp["price"] = pd.Series(allp.index.map(last_price), index=allp.index).fillna(pd.Series(allp.index.map(_rev), index=allp.index).map(last_price)); allp["gap_value"] = allp.gap * allp.price
save("scrap_unbooked", allp[allp.moulded > 0].sort_values("gap", ascending=False).head(20).reset_index()[["part_no", "moulded", "bookings_lost", "scrap_qty", "gap", "price", "gap_value"]], "Cast parts where castings lost between moulding and despatch exceed scrap booked: unbooked scrap, or castings despatched without a booking.")
foundry_scrap = SH["Work Centre Cell"].set_index("item").scrap_qty.reindex(["Fettling", "Die Shop", "ASF", "ASC", "Greensand"]).sum()
save("scrap_summary", [{"scrap_booked": SH["Scrap Reason"].scrap_qty.sum(), "scrap_per_year": SH["Scrap Reason"].scrap_qty.sum() / YEARS_SC, "scrap_pct": 0.88, "scrap_value_priced": sp_.scrap_value.sum(), "scrap_value_per_year": sp_.scrap_value.sum() / YEARS_SC, "priced_share_of_scrap_pct": 100 * sp_[sp_.price.notna()].scrap_qty.sum() / sp_.scrap_qty.sum(), "top10_parts_share_pct": 100 * sp_.scrap_qty.nlargest(10).sum() / sp_.scrap_qty.sum(), "top10_parts_value": sp_.sort_values("scrap_value", ascending=False).scrap_value.head(10).sum(),
                        "foundry_cells_scrap_booked": foundry_scrap, "bookings_castings_lost_same_period": oo.lost.sum(), "bookings_castings_moulded_same_period": oo.m.sum(), "bookings_loss_pct": 100 * oo.lost.sum() / oo.m.sum(), "foundry_scrap_pct_of_moulded": 100 * foundry_scrap / oo.m.sum(), "unexplained_castings": oo.lost.sum() - foundry_scrap, "parts_with_scrap": len(sp_)}], "Scrap totals and reconciliation to shop-floor bookings.")
save("scrap_value_parts", sp_.sort_values("scrap_value", ascending=False).head(15)[["item", "customer", "scrap_qty", "scrap_pct_of_booked", "price", "scrap_value"]].rename(columns={"item": "part_no"}), "Parts with the highest value of booked scrap at recent selling price.")

# ===== I. where to look
_c = pl[(pl.yr >= 2025) & pl.ckey.notna() & (pl.ordered > 0) & (pl.order_net > 0)].assign(up=lambda x: x.order_net / x.ordered).sort_values("up"); avg_cast_price = float(_c.up.iloc[np.searchsorted(_c.ordered.cumsum().values, _c.ordered.sum() / 2)])
unp = so[(so.status != "Cancelled") & (so.kind == "Product") & (so.ordered > 0) & (so.order_net == 0) & (so.yr == 2026)]
K = {r["key"]: r["value"] for r in J("kpis")}; y26 = {r["group"]: r for r in J("y26_yield_line")}["All lines"]
loss26 = 1 - y26["yield_pct_janjul"] / 100; lossall = 1 - y26["yield_pct_all"] / 100; m26 = mo[mo.date.dt.year == 2026].qty.sum() * 365 / WEND.dayofyear
num = lambda v: float(v) if v is not None else 0.0
W = [
 {"where": "Open orders past their required date", "evidence": f"{len(od)} lines, {od.customer.nunique()} customers", "value_gbp": od.os_net.sum(), "basis": "Outstanding order value, one-off", "next_step": "Work the overdue list by customer; confirm which have shipped but not been invoiced."},
 {"where": "Order book due in the next 90 days", "evidence": f"{int(((ob.days_to_required >= 0) & (ob.days_to_required <= 90)).sum())} lines", "value_gbp": ob[(ob.days_to_required >= 0) & (ob.days_to_required <= 90)].os_net.sum(), "basis": "Outstanding order value, one-off", "next_step": "Load against line capacity in standard hours; ASF is the tight line."},
 {"where": "Casting yield in 2026 (ASC, ASF, Airset)", "evidence": f"Loss {100 * loss26:.1f}% Jan to Jul 2026 against {100 * lossall:.1f}% long run", "value_gbp": (loss26 - lossall) * m26 * avg_cast_price, "basis": f"Extra castings lost per year at £{avg_cast_price:.0f}, the typical cast-part price", "next_step": "Rerun Scrap Analysis by month for 2026; stock-check the ten worst parts."},
 {"where": "Prices that have not followed wages", "evidence": f"{len(behind)} parts worth £{behind.value_annual.sum():,.0f} a year", "value_gbp": behind.shortfall_if_price_matched_wages.sum(), "basis": "Annual revenue if price had tracked the wage floor", "next_step": "Reprice at next order, largest shortfall first."},
 {"where": "Booked scrap", "evidence": f"{SH['Scrap Reason'].scrap_qty.sum():,.0f} units since Oct 2021; 70% cast defect", "value_gbp": sp_.scrap_value.sum() / YEARS_SC, "basis": "Per year at selling price", "next_step": "Defect review on the ten highest-value parts; Die Shop and Fettling book 60% of it."},
 {"where": "Castings lost but not booked as scrap", "evidence": f"{oo.lost.sum():,.0f} lost on bookings against {foundry_scrap:,.0f} scrapped in foundry cells", "value_gbp": max(oo.lost.sum() - foundry_scrap, 0) * avg_cast_price / YEARS_SC, "basis": "Per year at typical cast-part price; upper bound", "next_step": "Make a scrap booking mandatory to close a works order short."},
 {"where": "Finished castings not booked to despatch", "evidence": f"{num(K.get('buf_fettle')):,.0f} castings, last 90 days", "value_gbp": num(K.get("buf_fettle")) * avg_cast_price, "basis": "Stock or unbilled sales at typical cast-part price, one-off", "next_step": "Count and reconcile to invoices since August 2026."},
 {"where": "Castings waiting between moulding and fettling", "evidence": f"{num(K.get('buf_mould')):,.0f} castings, last 90 days", "value_gbp": num(K.get("buf_mould")) * avg_cast_price, "basis": "Work in progress at typical cast-part price, one-off", "next_step": "Cap the queue and fettle first in, first out."},
 {"where": "Order lines in 2026 with no price", "evidence": f"{len(unp)} lines, {unp.ordered.sum():,.0f} units", "value_gbp": (unp.ordered * unp.key.map(last_price)).sum(), "basis": "At the part's recent selling price; may be free-of-charge replacements", "next_step": "Confirm each is a deliberate free issue."},
 {"where": "Cancelled order lines in 2026", "evidence": f"{len(canc[canc.yr == 2026])} lines, {canc[canc.yr == 2026].ordered.sum():,.0f} units", "value_gbp": (canc[canc.yr == 2026].ordered * canc[canc.yr == 2026].key.map(last_price)).sum(), "basis": "Estimated at recent selling price", "next_step": "Check which were re-entered under a new line and which were lost."}]
save("where_to_look", W, "Where to look in 2026, with the value attached to each. Values use selling price, not margin, and are not additive.")
save("link_quality", [{"sales_lines": len(so), "product_lines": len(pl), "sales_parts": pl.key.nunique(), "parts_matched_to_bookings": int(pl.key.drop_duplicates().isin(set(d.key)).sum()), "value_matched_pct": 100 * pl[pl.key.isin(set(d.key))].order_net.sum() / pl.order_net.sum(), "cast_parts_in_sales": int(pl.drop_duplicates("key").ckey.notna().sum()), "cast_value_pct": 100 * pl[pl.ckey.notna()].order_net.sum() / pl.order_net.sum(), "machined_value_pct": 100 * pl[pl.stream == "Sold under a machined part number"].order_net.sum() / pl.order_net.sum(), "cast_value_2026_pct": 100 * pl[pl.ckey.notna() & (pl.yr == 2026)].order_net.sum() / pl[pl.yr == 2026].order_net.sum(), "scrap_parts_matched_to_sales_pct": 100 * sp_.price.notna().mean(), "fifo_lines": len(est), "shipped_lines": tot_lines, "avg_cast_price_2026": avg_cast_price, "sales_from": so.entered.min(), "sales_to": END}], "How well the three datasets link by part number.")
so.drop(columns=["key", "ckey"]).to_csv(f"{OUT}/../sales_orders.csv", index=False)
print("report4 done", flush=True)
