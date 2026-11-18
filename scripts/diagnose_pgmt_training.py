"""Summarize the two archived Stage2 contracts and logged training windows."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/pgmt_reward_diagnosis_20261116'
OUT.mkdir(exist_ok=True)
RUNS = dict(old=ROOT/'logs/pgmt_stage2_center2x2/train_03',
            new=ROOT/'logs/pgmt_stage2_v8_emd/train_cuda0_20261112_104329')
report = {}
for name, path in RUNS.items():
    saved = torch.load(path/'model_19999.pt', map_location='cpu', weights_only=False)
    contract = saved['training_contract']
    env = contract['runtime']['environment']
    windows, sums, counts, digest = [], collections.defaultdict(float), collections.Counter(), hashlib.sha256()
    rows = []
    for i, line in enumerate((path/'progress.jsonl').open('rb')):
        digest.update(line)
        r = json.loads(line)
        t = r['telemetry']
        for k, v in t.items():
            if k.startswith(('RewardRate/', 'RewardGroupStep/', 'TrackingSnapshot/', 'Sampler/', 'TerrainMix/', 'TerrainReward/', 'TerminationCount/', 'OutcomeWindow/')) or (k.startswith('Curriculum/') and k.endswith('mean_level')):
                sums[k] += v
                counts[k] += 1
        if i == 0:
            columns = [k for k in t if k.startswith(('RewardRate/', 'RewardGroupStep/', 'TrackingSnapshot/', 'Sampler/'))]
        rows.append([t[k] for k in columns])
        if (i+1) % 1000 == 0:
            values = {k: v/counts[k] for k,v in sums.items()}
            families = sorted({k.split('/')[1] for k in sums if k.startswith('OutcomeWindow/')})
            outcomes = {f: {kind: sum(v for k,v in sums.items() if k.startswith('OutcomeWindow/'+f+'/') and k.endswith('/'+kind))
                       for kind in ('success','failure','map_exit','timeout','incomplete_motion','other_truncation')} for f in families}
            windows.append(dict(start=i-999,end=i,means=values,outcomes=outcomes,
                                success_by_family={f: o['success']/max(1,sum(o.values())) for f,o in outcomes.items()}))
            sums.clear(); counts.clear()
    np.savez_compressed(OUT/f'{name}_training_series.npz',columns=columns,values=np.asarray(rows))
    report[name] = dict(run=str(path),progress_sha256=digest.hexdigest(),iterations=i+1,
        rewards=env['rewards'],tracking=env['pgmt_tracking'],contacts=env['pgmt_contact_events'],
        touchdown=env['pgmt_touchdown_quality'],relaxation=env['pgmt_tracking_relaxation'],
        algorithm=contract['algorithm'],steps_per_update=contract['num_steps_per_env'],
        command={k:v for k,v in env['commands']['motion'].items() if k not in ('motion_files','body_names')},windows=windows)
    for w in (windows[0],windows[-1]):
        print(name,w['start'],{k:round(v,5) for k,v in w['means'].items() if k.startswith('RewardRate/terrain_')},w['success_by_family'],flush=True)

emd=[]
for p in sorted((RUNS['new']/'emd_evaluations').glob('*/result.json'),key=lambda p:int(p.parent.name.split('_')[1])):
    r=json.loads(p.read_text()); a=np.asarray(r['emd']); names=r['motion_files']
    walking=np.asarray([Path(n).stem.startswith('walk') for n in names])
    weights=2**(2*np.clip(a,.5,2))
    emd.append(dict(step=r['global_step'],mean=float(a.mean()),walk_mean=float(a[walking].mean()),
                    walk_sampling_probability=float(weights[walking].sum()/weights.sum()),
                    straight_clip=float(a[[Path(n).stem for n in names].index('walk1_subject1_clip0')]),
                    saturated_high_count=int((a>=2).sum())))
report['emd_series']=emd
(OUT/'training_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print('EMD first/last',emd[0],emd[-1])
