"""Read-only, reproducible comparison of the three completed PGMT Stage2 runs."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pgmt-analysis-mpl')
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent
OUTCOMES = ['success', 'failure', 'map_exit', 'timeout', 'incomplete_motion', 'other_truncation']
RUNS = ['train_01', 'train_02', 'train_03']
LABELS = ['01: event T1, w=10', '02: continuous T1, w=10', '03: continuous T1, w=1.5']
COLORS = ['#72869c', '#d85c48', '#168977']


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b != 0)


def finite_json(x):
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, np.generic):
        return finite_json(x.item())
    return None if isinstance(x, float) and not np.isfinite(x) else x


def contract_diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((contract_diff(a.get(k), b.get(k), f'{path}.{k}') for k in sorted(set(a) | set(b))), [])
    return [] if a == b else [dict(path=path, before=a, after=b)]


def read_run(name):
    run = BASE / name
    checkpoint = run / 'model_19999.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    contract = saved['training_contract']
    env = contract['runtime']['environment']
    course = saved['environment_state_dict']['curriculum']
    families = course['identity']['families']
    n, nf, nl = saved['next_iteration'], len(families), course['counts'].shape[1]
    transitions = env['scene']['num_envs'] * contract['num_steps_per_env']
    dt = env['sim']['dt'] * env['decimation']
    counts = np.empty((n, nf, nl, len(OUTCOMES)), dtype=np.int64)
    population = np.empty((n, nf, nl), dtype=np.int64)
    cumulative = np.zeros_like(counts[0])
    count_keys = [f'OutcomeWindow/{f}/L{l}/{o}' for f in families for l in range(nl) for o in OUTCOMES]
    cum_keys = [f'Curriculum/{f}/L{l}/{o}' for f in families for l in range(nl) for o in OUTCOMES]
    pop_keys = [f'Curriculum/{f}/L{l}/env_count' for f in families for l in range(nl)]
    columns = None
    data = None
    digest = hashlib.sha256()
    error = 0.
    with (run / 'progress.jsonl').open('rb') as stream:
        for i, line in enumerate(stream):
            digest.update(line)
            record = json.loads(line)
            assert record['iteration'] == i and record['completed_iteration'] == i + 1 and i < n
            t = record['telemetry']
            assert t['Window/transitions'] == transitions
            counts[i] = np.array([t[k] for k in count_keys]).reshape(nf, nl, -1)
            population[i] = np.array([t[k] for k in pop_keys]).reshape(nf, nl)
            cumulative += counts[i]
            assert np.array_equal(cumulative, np.array([t[k] for k in cum_keys]).reshape(nf, nl, -1))
            assert cumulative.sum() == record['reward_group_audit']['done_events']
            values = {k: v for k, v in t.items() if k.startswith(('RewardStep/', 'RewardGroupStep/',
                      'TrackingSnapshot/', 'TerminationCount/', 'Sampler/')) or k.endswith('/mean_level')}
            values.update({f'Loss/{k}': v for k, v in record['losses'].items()})
            values.update({f'ValueHeadGradient/{k}': v for k, v in record['value_head_gradient_l1'].items()})
            for k in ['learning_rate', 'collection_s', 'update_s', 'valid_patch_fraction', 'coverage_mean',
                      'center_boundary_fraction', 'center_overlap_fraction', 'center_overlap_area_fraction',
                      'peak_allocated_mib', 'peak_reserved_mib']:
                values[k] = record[k]
            for k in ['terrain_attention_per_head', 'center_std_m']:
                values.update({f'{k}/{j}': v for j, v in enumerate(record[k])})
            if columns is None:
                columns = list(values)
                data = np.empty((n, len(columns)))
            data[i] = [values[k] for k in columns]
            error = max(error, abs(sum(t['RewardGroupStep/'+g] for g in env['pgmt_reward_groups']) - t['RewardStep/total']))
            for f, family in enumerate(families):
                expected = population[i, f] @ np.arange(nl) / population[i, f].sum()
                assert abs(t[f'Curriculum/{family}/mean_level'] - expected) < 2e-6
            if (i + 1) % 5000 == 0:
                print(f'{name}: {i+1}/{n} verified', flush=True)
    assert i + 1 == n and np.isfinite(data).all() and (counts >= 0).all()
    assert np.array_equal(cumulative, course['counts'].numpy())
    assert (population.sum((1, 2)) == env['scene']['num_envs']).all()
    assert (population.sum(2) == population[0].sum(1)).all()
    assert error < 1e-10
    index = {c: j for j, c in enumerate(columns)}

    def window(a, b):
        means = {c: float(data[a:b, j].mean()) for j, c in enumerate(columns)}
        c = counts[a:b].sum(0)
        cf = c.sum(1)
        family_ends = cf.sum(-1)
        return dict(start=a, end=b-1, means=means, outcomes_by_level=c,
                    outcomes_by_family=cf, success_by_family=dict(zip(families, ratio(cf[:, 0], family_ends))),
                    success_by_level=ratio(c[..., 0], c.sum(-1)), ended_by_level=c.sum(-1),
                    success_all=float(cf[:, 0].sum()/cf.sum()),
                    mean_level_by_family={f: means[f'Curriculum/{f}/mean_level'] for f in families},
                    t1_rate_mean=means['RewardStep/terrain_touchdown_quality']/(dt*env['rewards']['terrain_touchdown_quality']['weight']),
                    t2_score_mean=means['RewardStep/terrain_reference_contact_match']/(dt*env['rewards']['terrain_reference_contact_match']['weight']))

    windows = [window(a, min(a+1000, n)) for a in range(0, n, 1000)]
    prob = np.load(run/'motion_sampling_probabilities.npy')
    names = course['identity']['motion_names']
    assert len(prob) == len(names) and abs(prob.sum()-1) < 1e-5
    top = np.argsort(prob)[-10:][::-1]
    summary = dict(run=str(run), checkpoint=str(checkpoint), iterations=n, num_envs=env['scene']['num_envs'],
                   steps_per_iteration=contract['num_steps_per_env'], transitions=transitions*n, dt=dt,
                   status=json.loads((run/'training_status.json').read_text())['status'],
                   launch=json.loads((run/'launch.json').read_text()),
                   source_checkpoint_sha256=json.loads((run/'training_status.json').read_text())['checkpoint_sha256'],
                   training_walltime_s=json.loads((run/'training_status.json').read_text())['training_walltime_s'],
                   progress_sha256=digest.hexdigest(), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   families=families, reward_weights={k:v['weight'] for k,v in env['rewards'].items()},
                   t1_config=env['pgmt_touchdown_quality'], first_window=windows[0], last_window=windows[-1], windows=windows,
                   final_mean_levels={f:float(data[-1,index[f'Curriculum/{f}/mean_level']]) for f in families},
                   total_outcomes=cumulative, final_population=population[-1],
                   top_sampling_probabilities=[dict(motion=names[j], probability=float(prob[j])) for j in top],
                   walk_sampling_probabilities=[dict(motion=s,probability=float(prob[j])) for j,s in enumerate(names) if s == 'walk1_subject1_clip1.npz'],
                   numerical_checks=dict(all_selected_metrics_finite=True, contiguous_iterations=True,
                      window_counts_match_cumulative=True, counts_match_checkpoint=True,
                      cumulative_done_events_match=True, population_and_mean_levels_match=True,
                      reward_group_sum_max_error=error),
                   last_window_minmax={c: [float(data[-1000:,j].min()),float(data[-1000:,j].max())]
                                      for j,c in enumerate(columns) if c.startswith('Loss/') or c=='learning_rate'})
    np.savez_compressed(OUT/f'{name}_arrays.npz',values=data,columns=np.array(columns),outcomes=counts,population=population)
    return summary, contract


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig, name):
        fig.savefig(OUT/(name+'.png'),dpi=170)
        fig.savefig(OUT/(name+'.svg'))
        plt.close(fig)
    families = results['train_03']['families']
    for metric,filename,title in [('success_by_family','01_terrain_success','Success / all ended episodes (%)'),
                                  ('mean_level_by_family','02_terrain_levels','Mean assigned curriculum level')]:
        fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for ax,f in zip(axs.flat,families):
            for run,color,label in zip(RUNS,COLORS,LABELS):
                w=results[run]['windows'];x=[v['end']+1 for v in w]
                y=[v[metric][f]*(100 if metric=='success_by_family' else 1) for v in w]
                ax.plot(x,y,color=color,label=label,lw=2)
            ax.set(title=f,xlabel='Completed PPO updates');ax.grid(alpha=.2)
            if metric=='mean_level_by_family':ax.set_ylim(0,9)
        axs.flat[-1].axis('off');handles,labels=axs.flat[0].get_legend_handles_labels()
        axs.flat[-1].legend(handles,labels,loc='center',frameon=False)
        fig.suptitle(title+' | non-overlapping 1,000-update windows')
        save(fig,filename)
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    metrics=[('TrackingSnapshot/upper_body_orientation_rad','Upper-body orientation error (rad)',1),
             ('TrackingSnapshot/lower_body_position_rmse_m','Lower-body position RMSE (m)',1),
             ('RewardStep/terrain_slip','Slip penalty magnitude',-1),
             ('RewardStep/auxiliary_action_rate','Action-rate penalty magnitude',-1),
             ('TerminationCount/anchor_ori','Orientation term triggers / update',1),
             ('t2_score_mean','T2 reference contact match score',1)]
    for ax,(k,title,sign) in zip(axs.flat,metrics):
        for run,color,label in zip(RUNS,COLORS,LABELS):
            w=results[run]['windows'];x=[v['end']+1 for v in w]
            y=[sign*(v[k] if k=='t2_score_mean' else v['means'][k]) for v in w]
            ax.plot(x,y,label=label,color=color,lw=2)
        ax.set(title=title,xlabel='Completed PPO updates');ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle('Tracking and stability proxies | changing training distribution, no fixed-policy evaluation')
    save(fig,'03_tracking_stability')
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,f in zip(axs.flat,families):
        fi=families.index(f)
        for run,color,label in zip(RUNS,COLORS,LABELS):
            w=results[run]['last_window'];y=w['success_by_level'][fi]*100;cnt=w['ended_by_level'][fi]
            ax.plot(range(10),y,color=color,label=label,lw=2)
            good=cnt>=100;sparse=(cnt>0)&~good
            ax.scatter(np.arange(10)[good],y[good],color=color,s=20)
            ax.scatter(np.arange(10)[sparse],y[sparse],edgecolor=color,facecolor='white',s=35,zorder=4)
        ax.set(title=f,xlabel='Birth-assigned level',ylim=(-3,103),xticks=range(10));ax.grid(alpha=.2)
    axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('Last 1,000 updates: success (%) by level | hollow markers: <100 ended episodes')
    save(fig,'04_success_by_level')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for run,color,label in zip(RUNS,COLORS,LABELS):
        w=results[run]['windows'];x=[v['end']+1 for v in w]
        axs[0].plot(x,[v['means']['RewardStep/terrain_touchdown_quality'] for v in w],color=color,label=label)
        axs[1].plot(x,[v['means']['Sampler/effective_motion_count'] for v in w],color=color,label=label)
    for ax,title in zip(axs,['T1 weighted reward per control step','Sampler effective motion count (1 / sum p^2)']):
        ax.set(title=title,xlabel='Completed PPO updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'05_reward_and_sampler')


def numeric_diagnostics():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = {}
    for run in RUNS:
        with np.load(OUT/f'{run}_arrays.npz') as data:
            columns=data['columns'].tolist(); values=data['values']; idx={c:j for j,c in enumerate(columns)}
        loss=values[:,idx['Loss/value_function_auxiliary']]
        force=-values[:,idx['RewardStep/auxiliary_anchor_contact_force']]
        peak=int(19000+np.argmax(loss[19000:]))
        results[run]=dict(whole_quantiles=np.quantile(loss,[.5,.95,.99,.999,1]).tolist(),
            last1000_quantiles=np.quantile(loss[-1000:],[.5,.95,.99,.999,1]).tolist(),
            quantile_probabilities=[.5,.95,.99,.999,1], count_above_1=int((loss>1).sum()),
            count_above_10=int((loss>10).sum()),
            log_loss_log_anchor_force_correlation=float(np.corrcoef(np.log10(loss),np.log10(force))[0,1]),
            last1000_peak_iteration=peak,last1000_peak={c:float(values[peak,idx[c]]) for c in
                ['Loss/value_function_auxiliary','RewardStep/auxiliary_anchor_contact_force','learning_rate','Loss/approx_kl']})
        if run=='train_03':
            fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
            axs[0].semilogy(np.arange(len(loss))+1,loss,lw=.6,color='#287fac')
            axs[0].set(title='train_03: auxiliary value loss (all 20,000 updates)',ylabel='Loss (log scale)')
            axs[1].semilogy(np.arange(len(loss))+1,force,lw=.6,color='#c06a3b')
            axs[1].set(title='Anchor contact-force penalty magnitude',ylabel='Reward / control step (log scale)',xlabel='Completed PPO updates')
            for ax in axs:ax.grid(alpha=.2)
            fig.savefig(OUT/'06_auxiliary_loss_spikes.png',dpi=170)
            fig.savefig(OUT/'06_auxiliary_loss_spikes.svg')
            plt.close(fig)
    (OUT/'numerical_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')


def main():
    results,contracts={},{}
    for run in RUNS:
        results[run],contracts[run]=read_run(run)
        w=results[run]['last_window']
        print(run,json.dumps(finite_json({k:w[k] for k in ['success_by_family','mean_level_by_family','t1_rate_mean','t2_score_mean']})),flush=True)
    diffs={f'{a}_to_{b}':contract_diff(contracts[a],contracts[b]) for a,b in zip(RUNS,RUNS[1:])}
    output=dict(runs=results,contract_differences=diffs,aggregation='1,000-update means; episode rates use pooled counts; NaN cells become null',
                analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'comparison_summary.json').write_text(json.dumps(finite_json(output),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    plot(results)
    numeric_diagnostics()
    print('Saved',OUT,flush=True)


if __name__=='__main__':
    main()
