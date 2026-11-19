import pandas as pd, numpy as np, json, sys, os
d=pd.read_parquet(sys.argv[1]); R=json.load(open(sys.argv[2])); out=sys.argv[3]
here=os.path.dirname(os.path.abspath(__file__))
code={f:open(os.path.join(here,f)).read() for f in ['clean.py','prep.py','models.py','build_dash.py']}
def save(name, rows, files):
    meta=[{"label":f,"value":code[f],"type":"code","lang":"python"} for f in files]+[{"label":"Input","value":"WIP_REPORT_2021-2026.xlsx (Time Entry Details, 44,136 rows, 4 Jan 2021 to 19 Nov 2026)"}]
    json.dump({"rows":rows,"meta":meta},open(f"{out}/{name}.json","w"),allow_nan=False)
mc=d[d.hrs_clean>0]
ms=set(mc.work_centre)
# booked vs clock per person-day
tt=mc[mc.time_from!='00:00:00'].copy()
tt['f']=pd.to_timedelta(tt.time_from).dt.total_seconds()/3600; tt['t']=pd.to_timedelta(tt.time_to).dt.total_seconds()/3600
tt.loc[tt.t<tt.f,'t']+=24
def union(g):
    iv=sorted(zip(g.f,g.t)); tot=0; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    return tot+ce-cs
pdy=tt.groupby(['date','name']).apply(lambda g: pd.Series({'booked':g.hrs_clean.sum(),'clock':union(g)})).reset_index()
pdy['yr']=pdy.date.dt.year; bc=pdy.groupby('yr')[['booked','clock']].sum()
lt=d.groupby('wo_base').date.agg(['min','max']); lt['days']=(lt['max']-lt['min']).dt.days; lt['yr']=lt['min'].dt.year; lt=lt[lt.days>0]
rows=[]
for y in range(2021,2027):
    g=d[d.yr==y]; m=g[g.work_centre.isin(ms)]; h=g[g.hrs_clean>0].groupby('type').hrs_clean.sum()
    tot=h.sum()
    rows.append(dict(year=str(y),entries=int(len(g)),machining_entries=int(len(m)),machining_timed_pct=round(float((m.hrs_clean>0).mean()*100),1),
      booked_hrs=round(float(h.get('Booked',0)),0),setup_hrs=round(float(h.get('Setup',0)),0),lost_hrs=round(float(h.get('Lost',0)),0),
      setup_pct=round(float(h.get('Setup',0)/tot*100),1) if tot>50 else None, lost_pct=round(float(h.get('Lost',0)/tot*100),1) if tot>50 else None,
      booked_to_clock=round(float(bc.loc[y,'booked']/bc.loc[y,'clock']),2) if y in bc.index and bc.loc[y,'clock']>50 else None,
      mismatch_rows=int(g.flag_hrs_mismatch.sum()),mismatch_hrs=round(float(g.loc[g.flag_hrs_mismatch,'hrs'].sum()),0),
      wo_lead_median_days=float(lt[lt.yr==y].days.median()),wo_lead_p90_days=float(lt[lt.yr==y].days.quantile(.9)),
      fett_qty=int(g[g.work_centre=='FETT'].qty.sum()),desp_qty=int(g[g.work_centre.isin(['DESP','MDESP'])].qty.sum()),
      moulded_qty=int(g[g.work_centre.isin(['ASF','ASC','GSL','AIRSET SPURLINE','DIE'])].qty.sum())))
save('yearly',rows,['clean.py','prep.py','build_dash.py'])
wc=[{"work_centre":k,"total_hrs":v['total'],"setup_pct":v['setup_pct'],"lost_pct":v['lost_pct']} for k,v in R['wc_loss'].items()]
save('work_centres',wc,['clean.py','prep.py','models.py'])
save('lost_codes',[{"reason":k,"hrs":v} for k,v in R['lost_by_code'].items()],['clean.py','prep.py','models.py'])
save('batch',[{"batch_qty":k,"jobs":v['jobs'],"setup_share_pct":v['setup_share_pct'],"hrs_per_unit":round(v['hpu'],2)} for k,v in R['setup_share_by_batch'].items()],['clean.py','prep.py','models.py'])
F=R['forecast']
save('forecast',[
 {"method":"Gradient boosted model","mae_hrs":round(F['mae_model'],1),"within_25pct":round(F['within_25pct_model']*100,0)},
 {"method":"Work centre median","mae_hrs":round(F['mae_wc_overall_median'],1),"within_25pct":None},
 {"method":"Part median x qty","mae_hrs":round(F['mae_part_median'],1),"within_25pct":round(F['within_25pct_base']*100,0)}],['clean.py','prep.py','models.py'])
save('drivers',[{"factor":{"part_no":"Part number","qty":"Batch quantity","work_centre":"Work centre","op":"Operator","month":"Month"}[k],"importance":v} for k,v in sorted(F['importance'].items(),key=lambda x:-x[1])],['models.py'])
tp=[]
for p,v in R['top_parts_hrs'].items():
    r={"part_no":p,"hrs":v['hrs'],"qty":v['qty'],"jobs":v['jobs']}
    for y,x in R['top_parts_hpu'][p].items(): r['hpu_'+y]=None if x is None or (isinstance(x,float) and np.isnan(x)) else x
    tp.append(r)
save('top_parts',tp,['models.py'])
A=R['anomalies']
save('anomalies',[
 {"check":"Booked hours do not match clock times","entries":A['hrs_vs_clock_mismatch']['rows'],"hrs":round(A['hrs_vs_clock_mismatch']['hrs'])},
 {"check":"Person-days booked over 12 hrs","entries":A['person_days_over_12h'],"hrs":None},
 {"check":"Overlapping bookings, same person, different job","entries":A['overlapping_bookings'],"hrs":None},
 {"check":"Timed jobs booked with qty 0","entries":A['timed_qty0_booked'],"hrs":None},
 {"check":"Exact duplicate bookings","entries":A['duplicate_bookings'],"hrs":None},
 {"check":"Jobs far off their part norm (isolation forest)","entries":A['job_outliers']['n'],"hrs":None}],['prep.py','models.py'])
print(json.dumps(rows,indent=0)[:1500])
