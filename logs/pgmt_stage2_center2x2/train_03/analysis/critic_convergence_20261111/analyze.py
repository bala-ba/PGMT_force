"""Recompute critic convergence from every original train_03 progress record."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pgmt-critic-convergence-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
HEADS = ['upper', 'lower', 'terrain', 'auxiliary']
KEYS = ['value_function'] + ['value_function_' + h for h in HEADS]
rows, digest = [], hashlib.sha256()
with (RUN / 'progress.jsonl').open('rb') as stream:
    for i, line in enumerate(stream):
        digest.update(line)
        r = json.loads(line)
        assert r['iteration'] == i and r['completed_iteration'] == i + 1
        assert r['tensorboard_iteration'] == i
        row = [r['losses'][k] for k in KEYS]
        row += [r['learning_rate'], r['losses']['approx_kl'],
                r['telemetry']['RewardStep/auxiliary_anchor_contact_force']]
        rows.append(row)
        if (i + 1) % 5000 == 0:
            print(f'Read and checked {i + 1} iterations', flush=True)
data = np.array(rows)
assert len(data) == 20000 and np.isfinite(data).all()
columns = KEYS + ['learning_rate', 'approx_kl', 'anchor_force_reward']
x = np.arange(len(data))
total, aux, lr, force = data[:, 0], data[:, 4], data[:, 5], -data[:, 7]
head_average_error = float(np.abs(total - data[:, 1:5].mean(1)).max())
np.testing.assert_allclose(total, data[:, 1:5].mean(1), rtol=1e-6, atol=1e-7)

# Validate the original logs against the earlier extraction, rather than use it
# as the source of this new analysis.
with np.load(RUN / 'analysis/train_03_arrays.npz', allow_pickle=False) as old:
    old_columns = old['columns'].tolist()
    for j, name in enumerate(KEYS):
        np.testing.assert_array_equal(data[:, j], old['values'][:, old_columns.index('Loss/' + name)])

# Both log files consume the same loss_dict. Independently cross-check the first
# three iterations in the TensorBoard event file (simple_value is float32).
tb_checked, tb_max_error = set(), 0.0
event_file = next(RUN.glob('events.out.tfevents.*'))
for event in LegacyEventFileLoader(str(event_file)).Load():
    for value in event.summary.value:
        if event.step <= 2 and value.tag in ['Loss/' + k for k in KEYS]:
            j = KEYS.index(value.tag[5:])
            actual = value.simple_value
            expected = data[event.step, j]
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-8)
            tb_checked.add((event.step, value.tag))
            tb_max_error = max(tb_max_error, abs(actual - expected))
    # Other tags use wall-time or sample-count axes, so event.step itself
    # is not globally monotonic across this mixed-tag file.
    if len(tb_checked) == 15:
        break
assert len(tb_checked) == 15


def stats(v):
    return dict(mean=float(v.mean()), median=float(np.median(v)), std=float(v.std()),
                p05=float(np.quantile(v, .05)), p95=float(np.quantile(v, .95)),
                p99=float(np.quantile(v, .99)), minimum=float(v.min()), maximum=float(v.max()),
                count_gt_1=int((v > 1).sum()), count_gt_10=int((v > 10).sum()))


def window(a, b):
    return dict(start=a, end=b - 1, metrics={k: stats(data[a:b, j]) for j, k in enumerate(KEYS)},
                auxiliary_share_of_summed_loss=float(aux[a:b].sum() / data[a:b, 1:5].sum()))


blocks = [window(i, i + 1000) for i in range(0, len(data), 1000)]
peaks = np.argsort(total)[-10:][::-1]
spike_rows = [{"iteration": int(i), **{k: float(data[i, j]) for j, k in enumerate(columns)}}
              for i in range(19555, 19567)]
result = dict(
    run=str(RUN), source='progress.jsonl (all 20000 rows); TensorBoard first 3 iterations cross-checked',
    progress_sha256=digest.hexdigest(), records=len(data), finite=True, contiguous_iterations=True,
    tensorboard_crosscheck_scalars=len(tb_checked), tensorboard_max_abs_error=tb_max_error,
    total_vs_average_four_heads_max_abs_error=head_average_error, matches_existing_extraction=True,
    first_100=window(0, 100), first_1000=blocks[0], last_1000=blocks[-1],
    whole=window(0, len(data)), blocks_1000=blocks,
    median_change_first_to_last_percent={k: float(100 * (np.median(data[-1000:, j]) / np.median(data[:1000, j]) - 1))
                                         for j, k in enumerate(KEYS)},
    top_total_loss_peaks=[{"iteration": int(i), **{k: float(data[i, j]) for j, k in enumerate(columns)},
                           "auxiliary_fraction": float(aux[i] / data[i, 1:5].sum())} for i in peaks],
    log_auxiliary_loss_vs_log_anchor_force_correlation=float(np.corrcoef(np.log(aux), np.log(force))[0, 1]),
    late_spike_context=spike_rows,
    limitation='Training clipped value loss on changing rollout distributions; no held-out value targets or explained variance logged.',
)
(OUT / 'statistics.json').write_text(json.dumps(result, indent=2) + '\n')
np.savez_compressed(OUT / 'critic_series.npz', iteration=x, columns=np.array(columns), values=data)


def rolling(v, method='median', width=250):
    a = sliding_window_view(v, width)
    return x[width - 1:], (np.median(a, axis=1) if method == 'median' else a.mean(1))


plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
colors = ['#2563eb', '#d97706', '#059669', '#c026d3']
fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
ax = axs[0, 0]
ax.plot(x, total, color='#94a3b8', lw=.45, alpha=.65, label='Raw loss')
ax.plot(*rolling(total, 'mean'), color='#e07a25', lw=1.4, label='250-update mean')
ax.plot(*rolling(total), color='#172554', lw=1.6, label='250-update median')
ax.set(yscale='log', ylabel='Clipped value loss (log scale)', title='A  Total critic loss = average of 4 heads', xlabel='Iteration (0-based)')
ax.legend(fontsize=9, loc='upper right')
ax = axs[0, 1]
for j, (h, color) in enumerate(zip(HEADS, colors), 1):
    ax.plot(*rolling(data[:, j]), color=color, lw=1.7, label=h)
ax.set(yscale='log', ylabel='250-update rolling median (log scale)', title='B  Typical loss declines, then improves slowly', xlabel='Iteration (0-based)')
ax.legend(ncol=2, fontsize=9)
ax = axs[1, 0]
for j, (h, color) in enumerate(zip(HEADS, colors), 1):
    ax.plot(x[-1000:], data[-1000:, j], color=color, lw=.7, alpha=.85, label=h)
ax.set(yscale='log', ylabel='Head loss (log scale)', title='C  Last 1,000 updates: auxiliary spikes persist', xlabel='Iteration (0-based)')
ax.legend(ncol=2, fontsize=9, loc='upper right')
ax.annotate('19561: auxiliary = 107.65', (19561, aux[19561]), (19100, 35), fontsize=9,
            arrowprops={'arrowstyle': '->', 'color': '#555'})
ax = axs[1, 1]
sl = slice(19555, 19567)
ax.plot(x[sl], aux[sl], color=colors[3], marker='o', lw=1.8, label='Auxiliary value loss')
ax.set(yscale='log', ylabel='Auxiliary loss (log scale)', title='D  Example: transient loss and contact-force spike', xlabel='Iteration (0-based)')
ax.ticklabel_format(axis='x', style='plain', useOffset=False)
other = ax.twinx()
other.plot(x[sl], force[sl], color='#64748b', marker='s', ls='--', lw=1.4, label='|Torso contact penalty|')
other.set(yscale='log', ylabel='|Weighted torso contact penalty per step|')
other.spines['right'].set_visible(True)
lines = ax.get_lines() + other.get_lines()
ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc='upper right')
ax.text(.03, .73, 'LR: 1.48e-4 (19560) -> 2.25e-7 (19561)\nAuxiliary loss returns to 0.0318 by 19563',
        transform=ax.transAxes, fontsize=8.5, bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
for ax in axs.flat:
    ax.grid(alpha=.2, which='major')
fig.suptitle('PGMT Stage2 train_03 | 20,000 PPO updates | training critic convergence', fontsize=15)
fig.savefig(OUT / 'critic_convergence.png', dpi=180)
fig.savefig(OUT / 'critic_convergence.svg')
plt.close(fig)
print(json.dumps({k: result[k] for k in ['records', 'progress_sha256', 'tensorboard_crosscheck_scalars',
                                        'total_vs_average_four_heads_max_abs_error', 'median_change_first_to_last_percent']}, indent=2))
print('Saved:', OUT)
