"""Build audited numeric summaries and figures for the Stage2 reward investigation."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pgmt_diagnosis_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/pgmt_reward_diagnosis_20261116'
LABELS = dict(stage1='Stage1',old='Stage2 center2x2',new='Stage2 v8 EMD')
COLORS = dict(stage1='#198866',old='#3579b6',new='#d25a42')
training = json.loads((OUT/'training_analysis.json').read_text())
summary = dict(probes={},flat_plane={},matched_prefix={},audits={})
probes, rows = {}, []
for name in LABELS:
    p = OUT/name
    z = dict(np.load(p/'reward_probe_0000.npz'))
    probes[name] = z
    episodes = list(map(json.loads,(p/'episodes.jsonl').read_text().splitlines()))
    summary['probes'][name] = []
    for j, ep in enumerate(episodes):
        n = ep['control_steps']
        a = int(z['anchor_index'])
        error = np.linalg.norm(z['actual_body_pos'][:n,j,a,:2]-z['reference_body_pos'][:n,j,a,:2],axis=-1)
        rate = z['reward_rates'][:n,j]
        weighted = z['weighted_rates'][:n,j]
        contact = z['contact_contact'][:n,j]
        force = z['contact_force_norm'][:n,j]
        physics = z['physics_forces'][:n,j,:,0]
        physical_norm = np.linalg.norm(physics,axis=-1)
        weights=np.asarray([training['new']['rewards'][str(t)]['weight'] for t in z['term_names']])
        assert np.allclose(rate*weights,weighted,rtol=1e-5,atol=1e-5)
        assert np.allclose(physics[:,-1],z['contact_net_forces_w'][:n,j],atol=1e-4,rtol=1e-6)
        r = dict(model=name,family=ep['family'],success=ep['success'],seconds=ep['seconds'],
            xy_rmse_m=ep['xy_rmse_m'],max_torso_drift_m=float(error.max()),
            t1_full_and_drift_over_50cm_fraction=float(((rate[:,21]>.999)&(error>.5)).mean()),
            t2_when_drift_over_50cm=float(rate[error>.5,22].mean()) if (error>.5).any() else None,
            flight_fraction=float((~contact).all(-1).mean()),
            threshold_10_to_20N_fraction=float(((force>10)&(force<20)).mean()),
            sampled_force_squared=float((force**2).sum(-1).mean()),
            all_substep_force_squared=float((physical_norm**2).sum(-1).mean()),
            missed_contact_fraction=float(((physical_norm>10).any(1)&(force<=10)).mean()),
            weighted_rates={str(k):float(v) for k,v in zip(z['term_names'],weighted.mean(0))},
            raw_rates={str(k):float(v) for k,v in zip(z['term_names'],rate.mean(0))})
        summary['probes'][name].append(r)
        rows.extend(dict(model=name,family=ep['family'],term=str(k),raw_rate_mean=float(x),weighted_rate_mean=float(y))
                    for k,x,y in zip(z['term_names'],rate.mean(0),weighted.mean(0)))
with (OUT/'probe_reward_means.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

for j in range(7):
    n=int(min(probes['new']['control_steps'][j],probes['old']['control_steps'][j]))
    summary['matched_prefix'][summary['probes']['new'][j]['family']] = dict(seconds=n*.02, **{
        name:dict(terrain_rate=float(z['weighted_rates'][:n,j,-6:].sum(-1).mean()),
                  t5_rate=float(z['weighted_rates'][:n,j,-2].mean()),
                  total_rate=float(z['weighted_rates'][:n,j].sum(-1).mean())) for name,z in probes.items() if name!='stage1'})

flat_arrays, flat_episodes = {}, {}
for name in LABELS:
    p=OUT/('plane_'+name)
    status=json.loads((p/'training_status.json').read_text())
    assert status['status']=='completed' and status['policy_state_unchanged'] and status['source_checkpoint_unchanged']
    ep=list(map(json.loads,(p/'episodes.jsonl').read_text().splitlines()))
    data=np.load(p/'trajectories/batch_0000.npz')['xy']
    flat_arrays[name],flat_episodes[name]=data,ep
    row=list(csv.DictReader((p/'terrain_summary.csv').open()))[0]
    summary['flat_plane'][name]={k:float(row[k]) for k in ('success_rate','xy_rmse_m','strict_025_success_rate','strict_050_success_rate','same_tile_fraction')}
    summary['flat_plane'][name].update(trials=len(ep),successes=sum(r['success'] for r in ep))
    assert all(r['same_family_fraction']==1 and r['same_tile_fraction']==1 for r in ep)
    for j,r in enumerate(ep):
        x=data[:r['control_steps'],j].astype(float)
        assert abs(np.sqrt(((x[:,:2]-x[:,2:])**2).sum(-1).mean())-r['xy_rmse_m'])<1e-7
    protocol=json.loads((p/'protocol.json').read_text())
    if name=='stage1': first_protocol=protocol
    else:
        for key in ('motion_sha256','original_terminations','seed','initial_world_yaw_rad','start_frame','terrain_manifest_sha256'):
            assert first_protocol[key]==protocol[key],key
common=np.logical_and.reduce([np.isfinite(x).all(-1) for x in flat_arrays.values()])
all_success=np.logical_and.reduce([[r['success'] for r in ep] for ep in flat_episodes.values()])
summary['flat_matched_frames']=dict(frames=int(common.sum()),common_success_clips=int(all_success.sum()),models={})
for name,data in flat_arrays.items():
    squared=((data[...,:2].astype(float)-data[...,2:])**2).sum(-1)
    summary['flat_matched_frames']['models'][name]=dict(common_prefix_rmse_m=float(np.sqrt(squared[common].mean())),
        common_success_rmse_m=float(np.sqrt(squared[:,all_success].mean())))

initial=np.load(OUT/'initial_blind/trajectories/batch_0000.npz')['xy']
stage1=np.load(OUT/'stage1/trajectories/batch_0000.npz')['xy']
assert np.array_equal(initial,stage1,equal_nan=True)
summary['audits']['initial_blind_equals_stage1_all_recorded_xy']=True
body=np.load(OUT/'new_body_contacts/reward_probe_0000.npz')
body_force=np.linalg.norm(body['all_body_forces'][:,0],axis=-1)
summary['nonfoot_contact_flat']={str(name):dict(frames=int((body_force[:,i]>10).sum()),max_force_n=float(body_force[:,i].max()))
    for i,name in enumerate(body['body_names']) if 'ankle_roll' not in name and (body_force[:,i]>10).any()}
for j,n in enumerate(probes['new']['control_steps']):
    assert np.array_equal(body['actual_body_pos'][:n,j],probes['new']['actual_body_pos'][:n,j])
summary['audits']['body_force_probe_preserved_all_active_body_positions']=True
for name,previous in (('old','center2x2'),('new','v8_emd')):
    recorded=np.load(OUT/name/'trajectories/batch_0000.npz')['xy']
    reward_off=np.load(ROOT/'logs/pgmt_l9_straight_compare_20261116_final'/previous/'trajectories/batch_0000.npz')['xy']
    assert np.array_equal(recorded,reward_off,equal_nan=True)
    summary['audits'][name+'_native_rewards_on_off_identical_xy']=True
summary['t5_boundary']=json.loads((OUT/'t5_boundary_audit.json').read_text())

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
for name,z in probes.items():
    a=int(z['anchor_index']); t=np.arange(len(z['reward_rates']))*.02
    actual=z['actual_body_pos'][:,0,a];ref=z['reference_body_pos'][:,0,a]
    ax[0].plot(t,actual[:,0]-ref[:,0],label=LABELS[name],color=COLORS[name])
    error=np.linalg.norm(actual[:,:2]-ref[:,:2],axis=-1)
    ax[1].plot(t,error,color=COLORS[name])
    v=np.linalg.norm(z['actual_body_lin_vel'][:,0,a,:2],axis=-1)
    ax[2].plot(t,v,color=COLORS[name])
ax[2].plot(t,np.linalg.norm(z['reference_body_lin_vel'][:,0,a,:2],axis=-1),'k--',label='Reference speed')
for axis in ax:
    axis.axvspan(0,3,color='#ddd',alpha=.35);axis.grid(alpha=.2)
ax[0].legend(ncol=3);ax[0].set_title('Same walking clip: drift starts during the near-stationary opening')
ax[0].set_ylabel('Signed X error (m)');ax[1].set_ylabel('Torso XY error (m)');ax[2].set_ylabel('XY speed (m/s)');ax[2].set_xlabel('Time (s)');ax[2].legend()
fig.savefig(OUT/'flat_drift.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for name,z in probes.items():
    rates=z['weighted_rates'][:,0,-6:].mean(0);offset={'stage1':-.25,'old':0,'new':.25}[name]
    ax[0,0].bar(np.arange(6)+offset,rates,width=.24,label=LABELS[name],color=COLORS[name])
ax[0,0].set_xticks(range(6),['T1','T2','T3','T4','T5','T6']);ax[0,0].set_ylabel('Weighted reward rate (1/s)');ax[0,0].set_title('Flat walking clip: reward components');ax[0,0].legend(fontsize=8)
z=probes['new'];a=int(z['anchor_index']);error=np.linalg.norm(z['actual_body_pos'][:,0,a,:2]-z['reference_body_pos'][:,0,a,:2],axis=-1)
ax[0,1].scatter(error,z['t1_rate'][:,0],s=9,alpha=.3,color=COLORS['new']);ax[0,1].axvline(.5,color='k',ls='--');ax[0,1].set(xlabel='Torso XY drift (m)',ylabel='Raw T1 score',title='T1 can be perfect despite substantial drift')
for name in ('old','new'):
    ws=training[name]['windows'];x=[w['end']+1 for w in ws]
    ax[1,0].plot(x,[-w['means']['RewardRate/terrain_contact_switching'] for w in ws],label=LABELS[name],color=COLORS[name])
ax[1,0].set(xlabel='Training update',ylabel='T5 penalty magnitude (1/s)',title='Terrain reward gains mostly reduce T5 penalties');ax[1,0].legend(fontsize=8)
emd=training['emd_series'];x=np.asarray([e['step'] for e in emd])/(8192*24)
ax[1,1].plot(x,[e['mean'] for e in emd],label='862 motions');ax[1,1].plot(x,[e['walk_mean'] for e in emd],label='Walking motions')
ax[1,1].set(xlabel='Training update',ylabel='Joint-space EMD (rad distance)',title='Independent flat motion fidelity gets worse');ax[1,1].legend()
for axis in ax.flat:axis.grid(alpha=.15)
fig.savefig(OUT/'reward_evidence.png',dpi=160);plt.close(fig)

snapshot=OUT/'analysis_source';snapshot.mkdir(exist_ok=True)
paths=['scripts/eval_pgmt_xy.py','scripts/pgmt_reward_probe.py','scripts/diagnose_pgmt_training.py',
       'scripts/diagnose_pgmt_reward_results.py','pmt_tasks/pgmt_math.py','pmt_tasks/pgmt_contact_state.py',
       'pmt_tasks/pgmt_touchdown.py','pmt_tasks/pgmt_contact_match.py','pmt_tasks/pgmt_slip.py',
       'pmt_tasks/pgmt_stumble.py','pmt_tasks/pgmt_contact_switching.py','pmt_tasks/pgmt_contact_force.py',
       'pmt_tasks/pgmt_runtime.py','motion_tracking_rl/storage/rollout_storage.py']
summary['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
for p in paths:shutil.copy2(ROOT/p,snapshot/Path(p).name)
(OUT/'diagnosis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ('flat_plane','flat_matched_frames','matched_prefix','audits')},indent=2))
