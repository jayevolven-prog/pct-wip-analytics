# pct-wip-analytics
Precision Casting WIP time-entry pipeline. Drop a new WIP export into `data/` (newest .xlsx is used) and push: Railway reruns cleaning, models and the Postgres load.
Tables: time_entries, agg_*, model_runs. Views: machining_jobs, lost_time_by_month.
