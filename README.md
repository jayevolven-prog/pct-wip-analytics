# pct-wip-analytics
Precision Casting manufacturing analytics pipeline. Drop a new WIP export into `data/` (newest .xlsx is used) and push: Railway reruns cleaning, models and the Postgres load.

## Stages
`pipeline/stages.csv` maps operation numbers to shop-floor stages. New BOM scan points: add a row (op range, stage name, order, flow). Every flow, lead-time and process-mining output picks it up.

## Outputs (Postgres)
- `time_entries` cleaned entries; `agg_*` summary tables; `rpt_*` report tables (OEE, flow, survival, forecast, supply chain, capacity, quote model); `model_runs` JSON history.
- Views: `machining_jobs`, `lost_time_by_month`.

## Models
OEE (availability x performance x quality), chained Laspeyres productivity index, linear mixed model (batch elasticity, year effects), process mining (variants, directly-follows), Little's law, Kaplan-Meier and Cox survival, Holt-Winters forecast, PELT change points, ABC and Syntetos-Boylan demand classes, Herfindahl concentration, Kingman VUT queueing, LightGBM quantile regression with TreeSHAP, isolation forest anomalies.
