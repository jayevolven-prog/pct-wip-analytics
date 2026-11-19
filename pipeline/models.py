import pandas as pd, numpy as np, sys, json
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
d=pd.read_parquet(sys.argv[1]); R={}
mc=d[d.hrs_clean.notna()&(d.hrs_clean>0)].copy()
# 2026 machine shop presence
ms_centres=mc.work_centre.unique()
R['ms_rows_by_year']=d[d.work_centre.isin(ms_centres)].groupby('yr').agg(rows=('hrs','size'),timed=('hrs_clean',lambda s:(s>0).sum())).to_dict('index')
# ---- 1. hours per job forecast (machining, Booked) at WO-op level
b=mc[mc.type=='Booked']
job=b.groupby(['wo_no','part_no','work_centre']).agg(hrs=('hrs_clean','sum'),qty=('qty','max'),date=('date','min'),ops=('name','nunique'),op=('name','first')).reset_index()
job=job[job.qty>0]; job['yr']=job.date.dt.year; job['hpu']=job.hrs/job.qty
setup=mc[mc.type=='Setup'].groupby('wo_no').hrs_clean.sum()
job['setup_hrs']=job.wo_no.map(setup).fillna(0)
train=job[job.yr<=2023]; test=job[job.yr.isin([2024,2025])]
feats=['part_no','work_centre','op','qty','month']
for j in (train,test): j['month']=j.date.dt.month
X=lambda j: j[feats].assign(**{c:j[c].astype('category') for c in ['part_no','work_centre','op']})
cats=pd.concat([train,test])
for c in ['part_no','work_centre','op']:
    cat=pd.CategoricalDtype(cats[c].unique()); train[c]=train[c].astype(cat); test[c]=test[c].astype(cat)
m=HistGradientBoostingRegressor(categorical_features=[0,1,2],max_iter=300,learning_rate=0.05,random_state=0)
m.fit(train[feats],np.log1p(train.hrs))
pred=np.expm1(m.predict(test[feats]))
# baseline: part median hrs-per-unit from training * qty
pm=train.groupby('part_no',observed=True).hpu.median(); base=test.part_no.astype(str).map(pm.rename(index=str)).fillna(train.hpu.median())*test.qty
R['forecast']={'n_train':len(train),'n_test':len(test),'mae_model':mean_absolute_error(test.hrs,pred),'mae_part_median':mean_absolute_error(test.hrs,base),'mae_wc_overall_median':mean_absolute_error(test.hrs,test.work_centre.astype(str).map(train.groupby('work_centre',observed=True).hrs.median().rename(index=str)).fillna(train.hrs.median())),'median_job_hrs':test.hrs.median(),'within_25pct_model':float((abs(pred-test.hrs)<=0.25*test.hrs).mean()),'within_25pct_base':float((abs(base-test.hrs)<=0.25*test.hrs).mean())}
from sklearn.inspection import permutation_importance
pi=permutation_importance(m,test[feats],np.log1p(test.hrs),n_repeats=5,random_state=0)
R['forecast']['importance']=dict(zip(feats,np.round(pi.importances_mean,3)))
# batch size learning curve: hrs/unit vs qty within part (log-log slope, parts with >=5 jobs)
sl=[]
for p,g in job.groupby('part_no'):
    if len(g)>=5 and g.qty.nunique()>2: sl.append(np.polyfit(np.log(g.qty),np.log(g.hpu),1)[0])
R['batch_slope']={'n_parts':len(sl),'median_slope':float(np.median(sl)),'pct_negative':float(np.mean(np.array(sl)<0))}
# ---- 2. lost time and setup drivers
lt=mc[mc.type=='Lost']
R['lost_by_code']=lt.groupby('lost_time').hrs_clean.sum().sort_values(ascending=False).round(1).to_dict()
yy=mc.groupby(['yr','type']).hrs_clean.sum().unstack().fillna(0); yy['lost_pct']=yy.Lost/yy.sum(axis=1)*100; yy['setup_pct']=yy.Setup/yy[['Booked','Lost','Setup','Rework']].sum(axis=1)*100
R['type_by_year']=yy.round(1).to_dict('index')
wc=mc.groupby(['work_centre','type']).hrs_clean.sum().unstack().fillna(0); tot=wc.sum(axis=1)
wc['lost_pct']=wc.get('Lost',0)/tot*100; wc['setup_pct']=wc.get('Setup',0)/tot*100; wc['total']=tot
R['wc_loss']=wc[wc.total>500][['total','setup_pct','lost_pct']].sort_values('lost_pct',ascending=False).round(1).to_dict('index')
# setup hours vs batch size: setup share of job
js=job[job.setup_hrs>0]; js['setup_share']=js.setup_hrs/(js.setup_hrs+js.hrs)
js['qband']=pd.cut(js.qty,[0,2,5,10,25,50,10000],labels=['1-2','3-5','6-10','11-25','26-50','51+'])
R['setup_share_by_batch']=js.groupby('qband',observed=True).agg(jobs=('qty','size'),setup_share_pct=('setup_share',lambda s:round(s.median()*100,1)),hpu=('hpu','median')).to_dict('index')
# ---- 3. labour cost per unit trend (machining, Booked+Setup, rate>0)
c=mc[mc.rate_gbp_hr>0]; c['gbp']=c.hrs_clean*c.rate_gbp_hr
R['rate_by_year']=d[d.rate_gbp_hr>0].groupby('yr').rate_gbp_hr.agg(['median','mean']).round(2).to_dict('index')
pw=c[c.type.isin(['Booked','Setup'])].groupby(['part_no','yr']).agg(gbp=('gbp','sum'),hrs=('hrs_clean','sum')).reset_index()
q=job.groupby(['part_no','yr']).qty.sum().reset_index(); pw=pw.merge(q,on=['part_no','yr']); pw['hpu']=pw.hrs/pw.qty
top=job.groupby('part_no').hrs.sum().nlargest(10).index
R['top_parts_hpu']=pw[pw.part_no.isin(top)].pivot(index='part_no',columns='yr',values='hpu').round(2).to_dict('index')
R['top_parts_hrs']=job.groupby('part_no').agg(hrs=('hrs','sum'),qty=('qty','sum'),jobs=('wo_no','nunique')).loc[top].round(0).to_dict('index')
# ---- 4. anomalies
an={}
an['hrs_vs_clock_mismatch']={'rows':int(d.flag_hrs_mismatch.sum()),'hrs':float(d.loc[d.flag_hrs_mismatch,'hrs'].sum()),'top_people':d[d.flag_hrs_mismatch].groupby('name').hrs.agg(['size','sum']).sort_values('sum',ascending=False).head(5).round(0).to_dict('index')}
an['timed_qty0_booked']=int(((mc.type=='Booked')&(mc.qty==0)).sum())
an['timed_rate0']=int((mc.rate_gbp_hr==0).sum())
dup=d[d.hrs>0].duplicated(['date','name','wo_no','time_from','time_to'],keep=False); an['duplicate_bookings']=int(dup.sum())
# overlapping time for same person same day across different WOs
ov=0;ovh=0
tt=d[(d.hrs_clean>0)&(d.time_from!='00:00:00')].copy(); tt['f']=pd.to_timedelta(tt.time_from); tt['t']=pd.to_timedelta(tt.time_to)
for (dt,n),g in tt.groupby(['date','name']):
    g=g.sort_values('f'); e=g.t.shift(); o=(g.f<e)&(g.wo_no!=g.wo_no.shift())
    ov+=o.sum()
an['overlapping_bookings']=int(ov)
# person-day totals >12h
pdh=d.groupby(['date','name']).hrs_clean.sum(); an['person_days_over_12h']=int((pdh>12).sum())
# isolation forest on job hpu vs part norm
job['part_med']=job.groupby('part_no').hpu.transform('median'); job['ratio']=job.hpu/job.part_med
jj=job[job.groupby('part_no').hpu.transform('size')>=4].copy()
iso=IsolationForest(contamination=0.02,random_state=0).fit(np.c_[np.log(jj.ratio.clip(1e-3)),np.log1p(jj.qty),np.log1p(jj.hrs)])
jj['anom']=iso.predict(np.c_[np.log(jj.ratio.clip(1e-3)),np.log1p(jj.qty),np.log1p(jj.hrs)])==-1
an['job_outliers']={'n':int(jj.anom.sum()),'of':len(jj),'examples':jj[jj.anom].sort_values('ratio',ascending=False).head(8)[['wo_no','part_no','work_centre','qty','hrs','hpu','part_med']].round(2).to_dict('records')}
R['anomalies']=an
# ---- 5. other correlations
R['dow_hpu']=job.assign(dow=job.date.dt.day_name()).groupby('dow').agg(jobs=('hpu','size'),ratio_to_part_norm=('ratio','median')).round(3).to_dict('index')
R['multi_operator']=job.assign(multi=job.ops>1).groupby('multi').ratio.median().round(3).to_dict()
opn=job[job.groupby('op').op.transform('size')>=50].groupby('op').agg(jobs=('ratio','size'),ratio=('ratio','median')).sort_values('ratio').round(2)
R['operator_ratio']=opn.to_dict('index')
R['rework_hrs_by_year']=mc[mc.type=='Rework'].groupby('yr').hrs_clean.sum().round(1).to_dict()
# foundry throughput (qty) by year and centre
fd=d[d.work_centre.isin(['ASF','ASC','GSL','AIRSET SPURLINE','DIE','FETT','DESP','MDESP'])]
R['foundry_qty_by_year']=fd.groupby(['work_centre','yr']).qty.sum().unstack().fillna(0).astype(int).to_dict('index')
# lead time: first to last booking date per WO base
lt2=d.groupby('wo_base').date.agg(['min','max']); lt2['days']=(lt2['max']-lt2['min']).dt.days; lt2['yr']=lt2['min'].dt.year
R['wo_leadtime_days']=lt2[lt2.days>0].groupby('yr').days.agg(['size','median',lambda s:s.quantile(.9)]).round(0).to_dict('index')
json.dump(R,open(sys.argv[2],'w'),indent=1,default=str)
