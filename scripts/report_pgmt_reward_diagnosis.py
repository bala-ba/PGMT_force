"""Chinese evidence report for the recorded Stage2 regression investigation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/pgmt_reward_diagnosis_20261116'
d=json.loads((OUT/'diagnosis.json').read_text())
labels=dict(stage1='Stage1 · model_15000',old='Stage2 · center2x2',new='Stage2 · v8 EMD')
table=''.join(f'<tr><td>{labels[n]}</td><td>{r["successes"]}/862 ({r["success_rate"]:.1%})</td><td>{100*r["xy_rmse_m"]:.1f} cm</td><td>{r["strict_025_success_rate"]:.1%}</td><td>{r["strict_050_success_rate"]:.1%}</td></tr>' for n,r in d['flat_plane'].items())
prefix=''.join(f'<tr><td>{family}</td><td>{r["seconds"]:.2f} s</td><td>{r["old"]["terrain_rate"]:.3f}</td><td>{r["new"]["terrain_rate"]:.3f}</td><td>{r["old"]["total_rate"]:.3f}</td><td>{r["new"]["total_rate"]:.3f}</td></tr>' for family,r in d['matched_prefix'].items() if family in ('stairs_up','platforms'))
content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage2 平地退化与地形接触奖励诊断</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;color:#223047;max-width:1120px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}h2{{margin-top:38px;font-size:23px}}h3{{font-size:18px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:11px;border-bottom:1px solid #dce4ee;vertical-align:top}}th{{background:#eef3f8}}img{{width:100%;height:auto}}.lead{{background:#eef5fc;padding:20px;border-left:4px solid #3579b6}}.note{{font-size:14px;color:#53627a}}code{{background:#eef1f5;padding:2px 5px}}a{{color:#286baf}}li{{margin:8px 0}}.fig{{margin:28px 0}}</style>
<h1>Stage2 平地退化与 terrain-contact reward 诊断</h1><p class="note">2026-11-16 · Isaac / PhysX 原生推理与训练日志复核 · 全部模型只读，未修改训练奖励</p>
<div class="lead"><b>结论：</b>平地退化已经由纯平地全库对照证实。证据最强的机制是：静止时位置纠偏信号接近消失，XY 奖励死区大；地形接触目标允许“位置已错、接触仍高分”，T5 对快速换步的惩罚又偏强且有一帧边界偏差。Stage2 共享策略微调缺乏平地能力保留约束，随机短尾训练进一步削弱长期跟踪。不能把六项都称为已被 hack，也不能把 EMD 采样器当作直接奖励。</div>
<h2>1. 退化幅度与公平比较</h2>
<p>同一 862 个动作、每段从第 0 帧开始、最多 499 帧/9.98 秒、固定相同朝向与初始状态；大面积纯平地，100% 时间留在平地。关闭观测噪声、推力和动力学随机化，使用确定性 actor mean。三者采用相同 Stage2 终止规则，避免不同终止条件造成虚假的提升或退化。每个动作一次，这些百分比是动作集通过比例，不是多随机种子估计。</p>
<table><tr><th>模型</th><th>完整完成</th><th>XY RMSE</th><th>完成且全程 ≤25 cm</th><th>完成且全程 ≤50 cm</th></tr>{table}</table>
<p>只比较三者共同存活的 385,283 帧，XY RMSE 仍为 <b>25.8 / 32.5 / 31.0 cm</b>；共同完成的 668 个动作上为 <b>25.4 / 31.1 / 29.3 cm</b>。因此结论不是失败截断长度不同造成的。新 Stage2 的全库平均 XY 略好于旧 Stage2，但两者都比 Stage1 差。</p>
<p>此前视频片段 <code>walk1_subject1_clip0</code> 在 v8 平地上的 XY RMSE 为 <b>12.1 / 33.3 / 48.6 cm</b>。开头 3 秒参考几乎站定：Stage1 最大躯干漂移 3.8 cm，旧 Stage2 为 45.2 cm，新 Stage2 为 78.2 cm。随后参考开始走，两个 Stage2 才逐渐追回位置。</p>
<div class="fig"><img src="flat_drift.png" alt="平地同一动作的 X 偏差、XY 漂移和速度对照"></div>
<h2>2. 漂移的直接机制</h2>
<p><code>corrected_command</code> 的位置纠偏为 <code>clip(min(|v_ref|/0.5,1) × 0.5 × e_xy, ±0.5)</code>。参考静止时，位置反馈严格归零。actor 的 proprio 没有显式 XY 位置误差；位置误差主要经这个被门控的速度命令传入。因此在无限平地、参考速度为零且关节/姿态相同时，整体平移不会改变这些 actor 输入。近静止段参考平均速度仅约 0.0069 m/s，0.8 m 的偏移也只产生约 0.0055 m/s 的纠偏量。</p>
<p>多数身体位置/姿态奖励减去躯干位置并分别去除 yaw，整体 XY 偏移不会损失这些姿态分。唯一直接 XY 项为 <code>exp(-max(d-0.3,0)²/0.5²)</code>：30 cm 内满分，50 cm 时仍为 0.852。根速度奖励 σ=1 m/s，0.28 m/s 的单轴误差仍有约 96.2% 满分。原生 root_xy 终止又允许 1 m 偏移持续 0.2 s，普通“成功”远不等于位置准确。</p>
<p class="note">这些设计在 Stage1 已存在；它们解释为何 Stage2 改变步态后偏移缺乏足够反馈，不能单独证明 Stage2 必然退化。源码：<a href="../../pmt_tasks/pgmt_math.py">pgmt_math.py，83–110、162–203 行</a>。</p>
<h2>3. T1–T6 逐项判断</h2>
<table><tr><th>项 / 权重</th><th>实际机制、漏洞与本次证据</th><th>判断</th></tr>
<tr><td>T1 支撑质量<br>+1.5</td><td>目前是连续奖励，不是落脚事件奖励。要求两脚支撑模式与参考完全一致，再对参考支撑脚的 9 点高度平整度取均值。平地几何质量恒为 1，不检查 XY 跟踪、足端目标位置、足底相对姿态或承重是否充分。新模型整段 25.1% 的帧同时满足“躯干漂移 &gt;50 cm、T1 满分”。纯斜面即使足底贴合，也会因世界高度方差而扣分。严格模式匹配已阻止额外支撑脚取平均、重复落脚刷事件分等旧式漏洞。</td><td><b>已证实代理目标失配</b>；未发现通过落脚次数刷分。</td></tr>
<tr><td>T2 参考接触<br>+1.5</td><td>两脚逐脚比较力模长 &gt;10 N 与源动作接触标签，接触/不接触匹配都给分。没有轨迹、切向速度或承重质量约束，且台阶仍强制匹配原平地动作的同一帧节奏。新模型漂移 &gt;50 cm 的帧，T2 原始均分仍约 0.836。缺一只应支撑脚时，另一只正确摆动脚仍可拿半分。</td><td><b>已证实漂移仍高分</b>；卡阈值或用墙替代地面只属潜在漏洞。</td></tr>
<tr><td>T3 滑动<br>−1</td><td>惩罚接触脚的踝 link 原点世界 XY 速度平方，并非真实接触点切向滑速。正常脚掌滚动也可能被罚；无接触时不罚。单片段 Stage1→新 Stage2 的代价由 0.0547 降至 0.0248，而位置跟踪更差。说明少滑脚不能保证骨盆不漂；全身可通过换步发生位移。</td><td>有步态保守化与接触门控漏洞；<b>本次没有悬空逃罚证据</b>。</td></tr>
<tr><td>T4 绊碰<br>−20</td><td>条件为水平法向接触力 &gt; 5×|Fz|+10 N。它只检测特定碰撞形态，不能检测普通倾斜、慢漂或足部侧向摩擦。新模型平地以及多数失败地形前缀此项一直为 0。</td><td><b>检测覆盖不足</b>，不能视为整体稳定性指标；无主动刷取证据。</td></tr>
<tr><td>T5 快速切换<br>−30</td><td>任意脚接触状态切换，只要旧状态 age 太短就罚，不区分必要救步、正常参考短支撑或力阈值抖动。每次单脚事件扣 0.6 reward。已复现：真实 100 ms 支撑在第 15 步结束时，previous_age 只有 80 ms，仍被罚；120 ms 才免罚。参考标签 27,128 个有效相邻切换间隔中，1,570 个（5.79%）≤100 ms；这些源动作节奏也可能与 T5 冲突。</td><td><b>确定的边界偏差；最值得优先消融的接触项</b>。降低必要换步是合理解释，尚需训练消融定量归因。</td></tr>
<tr><td>T6 接触力<br>−10⁻⁶</td><td>所有脚的力平方，无正常承重免罚区，理论上倾向减小载荷/延长双支撑；只采控制周期最后一个物理子步，可能漏掉瞬时冲击。新模型平地平方力代价 0.0967，高于 Stage1 的 0.0771；没有通过降低脚部载荷获益。stairs_up 的末子步平方力均值约 7.74×10⁴ N²，四子步均值约 1.44×10⁵ N²，存在采样遗漏。</td><td><b>没有已观察到的减力投机</b>；有采样与承重目标设计风险。</td></tr></table>
<p>补充排除：新模型平地两脚同时无接触的比例为 0；足部处于 10–20 N 窄区间仅占脚×帧的 0.2%。记录到的非足部碰撞是左右手腕与髋部成对自碰撞，不是膝盖/手撑地分担体重。由此不支持“悬空、弱接触门槛、趴地卸载”是本次主要 hack。</p>
<h2>4. 奖励确实会给较差行为更高分</h2>
<p>在平地直走片段，把三个冻结策略都放入相同 Stage2 奖励函数：地形组平均 rate 为 <b>2.258 / 2.390 / 2.318</b>，而 XY RMSE 为 <b>12.1 / 33.3 / 48.6 cm</b>。接触分提高与位置能力退化同时发生。</p>
<p>以下只比较旧、新模型从相同起点到新模型失败时的<b>同一长度前缀</b>，避免用失败短片和成功全片直接比较。旧模型最终完成这两个地形，新模型都因 root_xy 失败；新模型的短期地形分及总分却都更高：</p>
<table><tr><th>地形</th><th>公共前缀</th><th>旧：地形 rate</th><th>新：地形 rate</th><th>旧：总 rate</th><th>新：总 rate</th></tr>{prefix}</table>
<p>stairs_up 的旧模型 T5 平均 rate 为 −1.132，新模型为 0；platforms 为 −0.533 对 0。该现象支持“规避换步/碰撞惩罚，可在失败前拿更好的短期分数”。它不等于已经证明完整折扣回报更高，也不等于证明 T5 是唯一原因。</p>
<div class="fig"><img src="reward_evidence.png" alt="接触奖励分解、T1 与漂移、T5 训练趋势及 EMD 曲线"></div>
<h2>5. 为什么 Stage2 会放大这些问题</h2>
<ol><li><b>两个 Stage2 的权重没有变化。</b> checkpoint 中 27 个奖励权重、T1 结算版本、接触阈值、tracking 和 relaxation 完全相同。代码常量中的 T1=10 被任务配置覆盖，训练实际为 1.5。</li>
<li><b>共享策略缺乏平地保留约束。</b> backbone_lr_scale=1，warmup_freeze_iters=0，value_only_warmup_iters=0，action_prior_loss_coef=0。四个 value head 的优势先相加再统一归一化，不能自动保护原来的平地步态。lower tracking 在台阶/斜坡/箱体高等级最多放宽 0.20 m / 0.70 rad / 0.50 rad；平地本身没有放宽，但共享参数仍受其他地形更新影响。Stage2 还从 end-effector acceleration 项中去掉了双脚，只保留手腕。</li>
<li><b>新训练的随机起点缩短有效回合。</b> 旧训练从第 0 帧开始，新训练从 0–497 均匀采样，然后到片段末尾 reset。理想无失败条件下平均剩余时长约 5 秒；每个片段前 3 秒的转移占比由约 30% 降到约 9%，而完成短尾也可升级地形。这会降低早期站稳与长期误差积累的训练覆盖。该 9% 是由采样规则推导，不是实际 rollout 统计。</li>
<li><b>EMD 不包含 XY，也不要求时序对齐。</b> 这是动作采样优先级，不是 actor 的直接奖励。对整条关节序列做最优指派，对帧重排及根部整体漂移不敏感，不能为静止漂移提供监督。独立平地 EMD 全库均值从 0.597 变为 0.917，walk 均值从 0.471 变为 0.752；后期训练成功率上涨，并不代表动作保真度恢复。</li>
<li><b>地形目标的提升主要来自减罚。</b> 新训练前/后各 1000 次更新，T5 加权 rate 从 −1.730 变为 −0.377；T1 为 0.577→0.575，T2 为 1.092→1.095。地形净分提升主要来自 T5 等惩罚减少。课程与采样分布同时变化，不能把这条日志曲线单独当作因果证明。</li>
<li><b>迁移时不保持原策略函数。</b> 初始 checkpoint 保留 Stage1 主干权重，但随机初始化的四个地形 token 立即进入 attention。相同片段的平地初始策略在 9.74 秒因 root_xy 失败。仅关闭这些 token 后，七地形全部已记录 XY 轨迹与 Stage1 <b>逐元素完全相同</b>。这说明参数复制并不等于行为保留。新模型后续在平地该片段能完成，故初始化问题是优化起点风险，不能直接解释所有最终失败。</li></ol>
<h2>6. 建议的修复与验证顺序</h2>
<ol><li><b>先补位置可观测性和静止纠偏。</b> 给 actor 显式局部 XY 误差，或保证静止时也存在有界位置反馈；缩小 30 cm 位置死区。修改输入/命令语义后需要重新训练或合适迁移，不能保证旧权重直接切换后变好。</li>
<li><b>修复 T5 的计时边界，再做降低/移除 T5 的小规模训练消融。</b> 区分阈值噪声、参考允许的快速换步和必要恢复步；可用接触迟滞与物理子步聚合。以相同数据、种子、训练步数同时检查平地位置和 L9 通行，避免只看总奖励。</li>
<li><b>保护 Stage1 的平地行为。</b> 采用初始等价的地形分支门控/残差，逐步开放；保留固定比例平地完整片段、站定动作和 Stage1 teacher 约束。单纯把新 token 值置零仍可能改变 softmax 分母，不能保证初始等价。</li>
<li><b>让接触分体现任务完成。</b> T1 增加落脚可行位置/足底贴合和承重信息；T2 允许合理相位容差，不把原平地接触时序作为台阶唯一正确答案；T3 用接触点切向滑速；T6 聚合四个物理子步并区分正常承重与冲击。</li>
<li><b>将训练成功与长程评估分开。</b> 保留从第 0 帧完整执行的固定评估集；同时记录普通成功、全程 XY≤25/50 cm、轨迹 RMSE、站定漂移、躯干速度/倾角、真实接触切换和峰值冲击。EMD 采样之外加入时间对齐误差与根部轨迹指标；短尾成功不应单独证明高等级长程能力。</li></ol>
<p>以上已定位可复现缺陷与奖励失配；若要给每项分配“导致多少退化”的百分比，需要配对重训练消融，本次没有把相关性写成该类因果结论。</p>
<h2>7. 审计与可复查资料</h2>
<p>原生 reward 开/关的旧、新模型轨迹与此前视频评估逐元素一致；接触记录保留 reset 前的终止帧，并记录全部 200 Hz 足部力子步。没有 PhysX 错误，冻结策略/归一化状态未变化。所有加权 reward rate 与 raw×weight 核对一致；RewardManager 每控制步只积分一次 dt。全库误差从原始 XY 数组独立复算；共同存活前缀和共同成功子集均复核了结论。</p>
<p class="note">仓库旧脚本 <code>analyze_pgmt_terrain_rewards.py</code> 仍把 T1 除以 dt² 并解释成“每秒落脚次数”，与当前 continuous support-pattern mean v2 不符；本报告不采用该单位解释。新的 rate 为加权每秒速率，事件次数另行计数。</p>
<ul><li><a href="diagnosis.json">诊断数据与审计</a> · <a href="training_analysis.json">两次训练契约与 1000-update 窗口</a> · <a href="probe_reward_means.csv">各地形 T1–T6 及其他项完整均值</a></li>
<li><a href="plane_stage1/episodes.csv">Stage1 纯平地 862 动作</a> · <a href="plane_old/episodes.csv">旧 Stage2 纯平地</a> · <a href="plane_new/episodes.csv">新 Stage2 纯平地</a></li>
<li><a href="new/reward_probe_0000.npz">新模型 7 地形逐帧/逐子步记录</a> · <a href="old/reward_probe_0000.npz">旧模型记录</a> · <a href="stage1/reward_probe_0000.npz">Stage1 记录</a></li>
<li><a href="t5_boundary_audit.json">T5 边界反例与参考接触统计</a> · <a href="../pgmt_l9_straight_compare_20261116_final/report.html">此前 L9 视频</a></li></ul>
<p class="note">代码及公式快照位于 analysis_source；各运行目录的 protocol.json 记录 checkpoint、动作、地形哈希和完整测试条件。5 m 地块出生评估保留在 flat_*，正式纯平地结论使用 plane_*，避免混入邻接地形。</p></html>'''
followup=OUT/'t1_switching_followup.json'
if followup.exists():
    f=json.loads(followup.read_text())
    rows=''.join(f'<tr><td>{labels[n]}</td><td>{sum(v["standing_opening"]["touchdown"])}</td><td>{v["standing_opening"]["t1_mean"]:.3f}</td><td>{sum(v["standing_opening"]["fast_switch"])}</td></tr>' for n,v in f.items())
    extra=f'''<h2>8. T1 持续给分与反复落脚：进一步核查</h2>
<p>参考要求双脚持续支撑的前 3 秒，两个 Stage2 都有 6 次重新落脚，但 T1 分数比 Stage1 低。反复落脚没有额外事件奖金，也没有提高该时段的 T1。</p>
<table><tr><th>模型</th><th>重新落脚次数（两脚合计）</th><th>T1 原始均分</th><th>T5 触发次数</th></tr>{rows}</table>
<p>准确的漏洞是：T1 不记住上一落脚位置或本次位移是否必要，只要再次满足接触模式且局部地形平整，就恢复满分。超过 T5 快速切换时间窗的调整步可以不受 T5 惩罚，累计产生全身漂移。它允许“短暂失分—在错误位置重新支撑—恢复满分”，但不能据此认定策略是为了增加 T1 而迈出这些步。</p>
<p>持续高均值不等于 PPO 梯度占比高。当前各组 GAE 先求和再归一化；若某项是可正确预测的常量，critic 的基线可以消除其大部分优势贡献。真正应比较的是各组优势方差、actor 梯度幅度与梯度冲突；变长回合、终止和 critic 误差又会使持续正奖励产生额外影响，不能简单认定它完全无效。现有日志没有直接给出逐项 actor 梯度。</p>
<p>建议将 T1 的几何安全目标与 T2 的接触时序目标分离，避免平地重复给支撑奖金；并把固定站定片段的重新落脚次数、落脚位移和位置漂移列为消融指标。将 T1 改为每次 touchdown 给奖金会引入“次数越多奖励越多”的新漏洞，应谨慎避免。</p>
<p><a href="t1_switching_followup.json">本次接触时序与足端位移统计</a></p>'''
    content=content.replace('</html>',extra+'</html>')
(OUT/'report.html').write_text(content)
print(OUT/'report.html')
