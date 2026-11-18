# InstantTrack 力/CoP Stage1 → Isaac Lab 四头 Stage2

Isaac 原生入口支持 `instanttrack.pgmt_force_cop.v1` 的权重迁移，以及
`--reference-run` 恢复历史环境与 PPO 配置。默认 stage2 配置保持原样。

## 使用 v6_v8difficulty、落脚事件 T1、新版 T5

在仓库根目录执行：

```bash
bash scripts/train_pgmt_stage2_native.sh \
  --reference-run logs/pgmt_stage2_center2x2/train_03 \
  --terrain-asset assets/terrain/pgmt_stage2_v6_v8difficulty \
  --contact-switching-mode geometry_v3 \
  --touchdown-event \
  --t1-t2-contact-threshold-n 1 \
  --initialize-from /home/lzp/projects/instantTrack/logs/g1_force_cop_20261116_135357/ckpt_025500.pt \
  --force-feedback-v1 --instanttrack-weights ema \
  --instanttrack-critic-init source \
  --device cuda:1 --seed 722 \
  --num_envs 8192 --max_iterations 20000 --save_interval 500 \
  --log_dir "logs/pgmt_stage2_force_isaac/v6v8_t1event_t2_1N_t5v3_$(date +%Y%m%d_%H%M%S)"
```

`--reference-run` 读取该目录的 `params/env.yaml`、`params/agent.yaml` 和
`initial.pt` 中的地形/动作/采样器身份，不导入该运行的训练权重或累计进度。
恢复后逐字段检查归档值，并在环境构建后再次比较 curriculum identity。
结果写入新运行的 `historical_config.json`。允许用 `--terrain-asset` 显式更换同布局的
地形难度版本，并记录新旧资产哈希；地形之外的 curriculum identity 必须仍与归档一致。
允许通过 `--touchdown-event` 和 `--contact-switching-mode` 分别覆盖 T1、T5，
通过 `--t1-t2-contact-threshold-n` 覆盖 T1/T2 的接触阈值，并在 `reward_overrides` 中记录模式、
权重及阈值；不允许同时覆盖整个 terrain reward 模式或 tracking 配置。

| 项目 | 恢复的 train_03 配置 |
| --- | --- |
| 地形 | 显式改为 `pgmt_stage2_v6_v8difficulty`，保留五类、十级、等比例和原 patch 布局；校验新 USD/mesh 哈希 |
| 动作库 | `pgmt_clipped_50hz_tcrs_v1`，862 个片段及原兼容性表 |
| 出生 | 中央 2×2 patch 内随机 XY、yaw ∈ [−π,π]、3 mm clearance、最多 128 次尝试 |
| 动作起点 | 固定第 0 帧，关闭 valid-support-start 门控 |
| 采样 | 按出生地形类别分别进行动作级 adaptive sampling。沿用 alpha=.001、beta=2、uniform_ratio=.1、update_interval=240、固定第 0 帧 |
| 课程 | 初始随机 L0/L1；完整动作成功升一级、失败降一级，边界/超时等独立结算 |
| 奖励 | 历史 27 项、四组；T1 落脚方差奖励、权重 1.5；T1/T2 阈值 1 N；T5 新版、权重 −0.05；T3 接触阈值仍为 10 N，原 tracking relaxation |
| PPO | 24 步、5 epoch、4 minibatch、初始 LR=1e-4，以及其余归档参数 |
| 物理/随机化 | 原 USD、50 Hz 控制、200 Hz 物理、原摩擦/恢复系数/推扰/零位/COM 配置 |

省略 `--terrain-asset` 时仍恢复原版 v6；上述命令明确选择修正难度的 `v6_v8difficulty`。
中央 2×2 指允许出生的 patch 集合，不是把机器人放在 patch 的固定中心点。

每个已结算回合按出生地形类别和起始动作片段累计成功与失败。每 240 个控制步，各类别独立更新
EMA 计数。在同一类别的兼容片段中，90% 的抽样权重按该类别的失败率平方分配，10% 均匀分配。
不同地形等级共享同类别的计数。跟踪失败记失败，已结算的无失败截断仍计入采样器的成功次数。
训练结束生成 `motion_sampling_by_terrain.npz`，其中 `families` 和 `motions` 给出行列顺序，
`eligible`、`failure_rate` 和 `probability` 为对应矩阵。原 `motion_sampling_probabilities.npy`
保存按环境地形占比加权的一维分布。

新 checkpoint 保存每类的 EMA 计数、当前窗口计数及更新步数。从旧全局采样 checkpoint
续训时，每个动作的历史计数和未结算窗口计数在兼容类别间等分，作为共同先验。
此后按实际出生类别记账。迁移情况记录在 `training_status.json` 的 `strict_restore.sampler`。

## 地形课程的普通回合生存比例与恢复概率

每个普通回合在结算时计算一次 `duration / horizon`。duration 是出生后实际经过的控制步数。
horizon 取出生帧到参考动作末帧之间的控制步数与环境时限的较小值，本命令的时限为 500 步（10 秒）。
例如，动作还剩 2 秒时，完整执行的生存比例为 1，执行 1 秒后摔倒的比例为 0.5。
首次初始化、手动重置或续训时中断的回合都不计入样本。

课程取最近 100 个普通回合的生存比例均值，样本不足时使用已有回合。
恢复调度概率从 0.10 开始，在第 20、40、60……个普通回合结算时复核。
每次目标为 `0.10 + 0.40 × 窗口均值`，当前概率向目标最多移动 0.02，到达目标即停止。
同一控制步有多个回合结束时按环境编号依次计入，跨过多个复核点就分别更新。

普通回合完整执行动作升一级，跟踪失败降一级。标记为 recovery 的回合单独记录成功、失败或截断，
沿用出生等级。当前命令使用普通 reset，仅维护恢复调度概率。
TensorBoard 和 `progress.jsonl` 中的 `Curriculum/ordinary_survival_last100` 是当前窗口均值，
`Curriculum/recovery_target_probability` 是上一次复核的目标，`Curriculum/recovery_probability` 是调度概率。
`Recovery/episodes/*` 记录恢复回合的各类结局。`Recovery/window_transition_fraction` 和
`Recovery/total_transition_fraction` 分别统计本轮和累计的恢复控制步数占全部环境控制步数的比例，
包含仍在进行的回合以及回合结束当步。使用当前命令时，这两个占比均为 0。

新 checkpoint 保存生存窗口、普通回合总数、目标与调度概率，以及 recovery 计数。
从旧 checkpoint 续训会恢复已有地形等级和历史结局，从续训后的第一个普通回合开始积累生存窗口。

## T1：每次落脚计算一次旧版高度方差

`--touchdown-event` 设置 `pgmt_touchdown_quality.settlement=touchdown_event_v1`，
并将 `terrain_touchdown_quality.weight` 固定为 **1.5**。省略该参数时保留历史连续 T1，
以兼容旧运行严格恢复。

- 落脚事件：地形过滤 Fz **>1 N** 建立支撑，之前至少连续 2 个控制步非支撑。
  每脚只在满足条件的落脚步计分；支撑保持、抬脚、reset 后的首次基线均不计分。
- 在落脚脚的 yaw 坐标系内取 **3×3** 个点，区域长 0.20 m、宽 0.10 m，中心前移 0.039 m。
  使用干净地形查询，保留原 torso/yaw 2 m 扫描范围有效性检查。
- 每脚质量 `q = exp(-Var(height[9]) / 0.03²)`，使用总体方差（除以 9）。
  任一点查询无效、越出扫描范围或非有限时，该脚本次奖励为 0，不延迟补发。
- 左右脚落脚质量直接求和，不依赖参考接触模式，也不除以参考支撑脚数。
- **保留旧版 rate 缩放**：`reward = 1.5 × control_dt × sum(q_touchdown)`，不额外除以 dt。
  当前 control_dt=0.02 s，平地单脚落脚一次 +0.03，双脚同落一次 +0.06。
- T5 继续使用独立的新版迟滞支撑流；T3/T4/T6 不变。

## T1/T2 接触阈值为 1 N

`--t1-t2-contact-threshold-n 1` 设置：

- T1：`pgmt_contact_events.support_threshold_n=1`，使用地形过滤的世界坐标 Fz，严格 **>1 N**。
- T2：`pgmt_reference_contact_threshold_n=1`，使用脚部全物体接触合力模长，严格 **>1 N**。
  使用独立判定，保留 T3 的共享 `contact_threshold_n=10`。
- T5 仍为物理子步平均 Fz 的 ≥1/≤0.5 N 迟滞；力/CoP 观测有效门槛保持不变。

阈值覆盖在历史配置恢复之后执行，并写入训练契约和 `historical_config.json`。
省略该参数时保持历史 T1/T2 阈值，便于旧 checkpoint 严格恢复。

## 独立更新 T5

`--contact-switching-mode geometry_v3` 只切换 T5；全局 `pgmt_terrain_reward_mode`
仍为 `legacy`。T1/T2 由上面的独立开关设置；T3/T4/T6 及出生、adaptive sampling、curriculum 保持历史配置。
省略该参数会使用历史 T5，便于旧运行严格恢复。

- 支撑力：每控制步内 4 个物理子步的 **地形过滤 Fz 均值**，不是步末瞬时力。
- 支撑状态：≥1 N 进入、≤0.5 N 离开，中间保持旧状态。
- 只在切换时计分：`q = max(0, 1 − 上一状态时长 / 0.1 s)`；
  初次观测建立基线，第一次切换建立已知时长的起点，不对未知初始时长计罚。
- 同方向的参考接触切换在 ±0.04 s 窗口内可豁免，每脚每次参考切换仅用一次。
- 原始 reward rate 为 `sum(q × 未豁免) / control_dt`；权重 **−0.05**，
  RewardManager 再乘 `control_dt`，因此每脚每事件实际惩罚为 `−0.05 × q`。
  80 ms 对应每脚 −0.01，100 ms 及以上为 0。
- 局部 reset 清空该环境支撑历史和已使用参考事件；终止步奖励保留冻结快照。

## 力反馈与迁移语义

- 从 InstantTrack Force checkpoint 新建 Isaac 训练时，`proprio[111]` 的前 99 维使用
  `g1_steady_env4096_20261108/ckpt_004000.pt` 对应的 99 维观测。按 G1 SDK
  关节顺序依次拼接相对带偏置默认姿态的关节角（29）、原始关节角速度（29）、
  pelvis 局部角速度（3）、投影重力（3）、上一步归一化动作（29）和 pelvis
  相对参考姿态误差（6，旋转矩阵前两列按行展开）。训练噪声的均匀半宽依次为
  0.01、0.5、0.2、0.05、0、0.05。
- `proprio_history[10,99]` 从最旧帧排到当前帧，共用当前观测的采样噪声和
  actor normalizer。`force_history[10,12]` 按左、右脚存储局部 Fx/Fy/Fz、
  CoP x/y、valid。两路历史逐帧投影后相加。当前帧 `proprio[99:111]` 直接使用
  同一控制步的双脚测量，与 `force_history[-1]` 逐元素一致。当前和历史使用相同的
  力与 CoP 固定缩放。完整 111 维用于查询历史，也与 128 维 intent 拼接为
  actor MLP 的 239 维输入。command 为 8×61。严格续训时按 checkpoint 元数据重建观测和网络。
- 使用原有显式地形过滤的左右脚 PhysX view。分别读取法向和摩擦的逐接触点数据，
  累加合力与相对 ankle link 的力矩，排除未列入训练地形的接触对象及自碰撞。
  两种接触流的接触点和 count/start 缓冲不同，不能混用。
- 足底参考平面 z=−.035 m，CoP 有效门槛及固定归一化尺度与 Stage1 V1 一致。
  无效 CoP 清零并设 valid=0，保留有效合力。每控制步末采样一次；局部 reset 清零
  当步反馈和该环境历史，重复读取不推进历史。每侧脚的 PhysX 共享缓冲区按
  `64 × 环境数` 分配；单脚可超过 64 点，按全局 count/start 完整聚合。
  真正的共享缓冲饱和、越界或非有限数据直接报错，不把截断结果送进策略。
- `--initialize-from` 指定训练权重，默认加载其中的 `ema`。
  迁移按关节名、标称默认关节角和动作尺度换算 90 个对应输入列及其归一化统计。
  新增的重力和 pelvis 姿态误差共 9 列，当前双脚反馈共 12 列，迁移时在
  当前帧投影和 actor 首层中将这些列置零，随后由 PPO 更新。
  对应 90 列继承源统计计数，重力和姿态误差 9 列从零累计。注意力、历史足力投影、command normalizer 和 actor
  后续层继承源权重。迁移核验在标称关节偏置下构造对应输入，将旧 torso
  姿态误差固定为源均值，关闭地形分支后比较动作和已复制价值头。
- Isaac 默认 `--instanttrack-critic-init source`。从命令的 Force EMA 权重迁移可对应的
  64 列（pelvis 局部线速度和角速度、29 个关节角与 29 个关节速度）、其归一化统计、
  128 列共享 intent 和后续隐藏层。新特权字段与地形 embedding 对应的 critic 首层权重从零学习。
  输出层源行 `[0,1,2]` 复制到四头目标行 `[0,1,3]`。
  即 body_tracking→upper、root_tracking→lower、regularization→auxiliary；
  新增 terrain 行（第 2 行，含权重和 bias）保留构造时的随机初始化。
  源与目标奖励组不同，这些参数作为训练初值使用；后续按 PMT 四组奖励更新。
  迁移审计将源模型独有的观测固定为源均值，在匹配输入上比较三个复制头的输出。
- PPO 优化器、采样计数和课程状态仍从新状态开始，不导入世界模型。
  初始探索标准差使用源 `action_noise=0.1`，不导入 BPTT 未使用的兼容 `std=1`。
- 新增地形分支使用原 PMT 初始化。四个价值头依次为 upper/lower/terrain/auxiliary。
  新 checkpoint 支持严格恢复；迁移来源、复制/初始化的参数名记录在 `migration.json`。
  部分复制的输出层另记录 `critic_head_mapping` 和 `initialized_rows`。
  只有显式选择 `--instanttrack-critic-init fresh` 才会重建整个 critic；mjlab 入口的默认行为保持不变。

## Critic 的 389 维输入

状态字段按 `g1_steady_env4096_20261108` 同次训练的 InstantTrack critic 定义排列，取当前帧。
原始的 10 帧、730 维未来参考窗口已从 critic 的直接输入去除。PGMT 的 8 帧 command 仍参与共享 IFM。

| Critic 首层切片 | 内容 |
| --- | --- |
| `[0:13]` | 实际 pelvis 离地高度、去 yaw 的姿态 6D、pelvis 局部线速度和角速度 |
| `[13:100]` | G1 SDK 顺序的实际关节角、实际关节角速度、当帧参考减实际的关节角，各 29 维 |
| `[100:142]` | 原 14 个跟踪刚体的位置误差。按实际 root XY 与相对 yaw 对齐参考身体后，转到实际 pelvis 的 yaw 坐标系 |
| `[142:146]` | 实际 pelvis 的 yaw 坐标系中，参考减实际的 XY 位置误差 2 维，随后是实际离地高度和参考离地高度 |
| `[146:152]` | 实际 pelvis 到参考 pelvis 的相对旋转，取矩阵前两列逐行展平 |
| `[152:185]` | 双脚碰撞形状的平均动态摩擦系数、29 个关节零位偏移和 torso 质心 xyz 偏移。依次按 `(μ−0.75)/0.45`、`偏移/0.01 rad`、`偏移/0.05 m` 换算，从启动随机化后的 PhysX 状态读取 |
| `[185:197]` | 当步双足局部力、CoP 和有效标记，复用 actor 同次测量和固定尺度 |
| `[197:261]` | 独立的干净地形图 MLP 输出，`441 → 128 → 64`，隐藏层用 ELU |
| `[261:389]` | 与 actor 共享的 128 维 intention fusion 输出 |

实际和参考高度均以米计，各用自身 pelvis 的世界 XY 查询碰撞地形，
取该 pelvis 的世界 Z 减去命中高度。参考取放置到训练地形后的当帧姿态。
实际高度出现在第 0 列及第 144 列。查询落在有效地形外时，该高度填 0。
地形图覆盖实际 torso 周围的 2 m × 2 m，随其 yaw 旋转，采样间距 0.1 m。
每格为地面世界 Z 减实际 torso 世界 Z，无效格填 0。Critic 对前 185 维更新独立的运行统计。

## 续训与短测

续训时保持原 reference-run、`--terrain-asset`、`--touchdown-event`、`--t1-t2-contact-threshold-n`、
`--contact-switching-mode`、环境数、device、seed 和初始学习率设置，把
`--initialize-from ...` 换成 `--resume 新运行/final.pt`，使用新的日志目录。
力反馈会从 checkpoint 自动识别，无需再次指定权重来源。
`--max_iterations` 是新增更新数；可以修改它和保存间隔。

短测可把环境数改为 32、更新数改为 2，并追加 `--force-feedback-audit`。
该选项用独立 NumPy float64 逐点计算复核正常力矩阵、局部合力、CoP、重复读取及
reset 历史，结果写入 `force_feedback_audit.json`。启用新版 T5 时，还逐物理子步独立
收集地形支撑力并在 CPU 重放迟滞、时长、参考豁免和带权奖励，写入
`contact_switching_audit.json`。启用落脚事件 T1 时，还独立复核落脚判定、PhysX 射线高度、
NumPy 方差和带权奖励，结果写入 `touchdown_event_audit.json`；正式训练省略该 CPU 审计选项。

学习率默认严格沿用 train_03 的 `1e-4`。当前 v6_v8difficulty/继承 critic 的
32 环境测试首轮 approximate KL=28.13，自适应 LR 降到 `1e-7`，第二轮 KL=.01765。
若选择更保守的迁移起始步长，可显式加
`--learning-rate 1e-6`，但这是一项相对 train_03 的配置变更，续训时也需保持该参数。
短测验证数据和训练链路，不代表原学习率最优或策略已经收敛。

## 2026-11-18 初版验证记录（原 v6、全新 critic）

- 相关单元/回归测试：80 passed、4 skipped（可选 GPU 集成项）；使用具备 mjlab
  依赖的 `mimiclite-mjlab` 环境运行。Isaac 数据链路另由以下真实 GPU 运行验证。
- [32 环境、2 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_smoke03/training_status.json)：
  源 actor 最大误差 0；归档配置与运行时 curriculum identity 均匹配；四个价值头、
  力投影和地形分支均有有限非零梯度。源 checkpoint 未改变。
- [严格恢复，第 2→6 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_resume01/training_status.json)：
  模型、normalizer、优化器、采样器/概率、课程、学习率及迭代号全部恢复通过。
- [真实接触审计](../logs/pgmt_stage2_force_isaac_validation/20261118_resume01/force_feedback_audit.json)：
  16 次快照，9016 个法向接触点、5907 个摩擦接触点，592 个有效 CoP，6 个 reset 行。
  最大合力误差 1.07e-4 N，CoP 误差 4.09e-8 m，正常力与传感器矩阵误差 8.37e-5 N。
- 未启动 8192 环境正式长训，也未以短测替代长期收敛或 sim-to-sim 性能评估。

## 2026-11-18 更新验证（v6_v8difficulty、继承 critic）

- 相关单元/回归测试：92 passed、4 skipped；增加了 EMA/model 两种来源的 critic
  隐藏层、normalizer、输出行拷贝及 terrain 行保持随机值的检查，以及地形覆盖不改变出生/采样配置的检查。
  另覆盖单脚 66 点、共享容量 128 点的完整合力/力矩聚合，避免把平均分配预算误当作单脚硬上限。
- [真实权重迁移审计](../logs/pgmt_stage2_force_isaac_validation/20261118_v6v8_critic03/critic_initialization_audit.json)：
  47 个完整张量和输出层三个复制行逐元素相等；critic normalizer count=78,401,536
  保持源值，新增 terrain 权重行 L2=0.6000225。关闭地形分支时 actor 和三个旧 critic 头的输出误差均为 0。
- [32 环境、2 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_v6v8_critic03/training_status.json)：完成；
  [地形覆盖报告](../logs/pgmt_stage2_force_isaac_validation/20261118_v6v8_critic03/historical_config.json)
  确认新地形哈希为 `bc2a2d153b662db76b74eae156274fbc8555084e4d0003db81bb4a25917ee95e`，
  其余历史 curriculum identity 匹配，源 checkpoint 未改变。
- [接触审计](../logs/pgmt_stage2_force_isaac_validation/20261118_v6v8_critic03/force_feedback_audit.json)：
  11 次快照，含 1 个 reset 行；最大合力误差 3.89e-4 N、CoP 误差 5.55e-8 m。
- [严格续训，第 2→4 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_v6v8_critic_resume03/training_status.json)：完成；
  模型、normalizer、优化器、采样器/概率、课程、学习率和迭代号恢复通过。
  该场景复现了单脚超过 64 点但共享缓冲未满的情况；已用全局区间聚合修复旧提取器的单脚限制。
  修复后的 10 次独立接触快照最大合力误差 1.15e-4 N、CoP 误差 4.46e-8 m。

## 2026-11-18 独立 T5 更新验证

- 相关单元/回归测试：147 passed、5 skipped（可选 GPU 测试）；覆盖旧 T5 回归、
  新版时长边界/时间步不变性/参考豁免，以及 T5 使用子步均值而 T2/T3 保留 10 N 阈值、
  不完整物理窗口失效和局部 reset 的终止步奖励缓存。
- [32 环境、2 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_t5_v3_smoke01/training_status.json)：
  完成，源 checkpoint 未改变，历史 curriculum identity 匹配；
  27 项奖励配置逐字段对比，只有 `terrain_contact_switching.weight` 从 −30 改为 −0.05，
  并新增独立 T5 模式；共享接触阈值和 tracking relaxation 保持原值。
- [独立 T5 审计](../logs/pgmt_stage2_force_isaac_validation/20261118_t5_v3_smoke01/contact_switching_audit.json)：
  48 个控制步、每步 4 个物理子步；272 次支撑切换、61 次惩罚事件、31 次参考豁免、
  1 个 reset 行。子步平均 Fz 最大误差 8.78e-5 N；CPU 重放的支撑状态、时长、
  豁免及 RewardManager 带权奖励全部匹配。
- [严格恢复检查](../logs/pgmt_stage2_force_isaac_validation/20261118_t5_v3_resume01/training_status.json)：
  从第 2 次更新恢复，新版 T5 模式和权重纳入训练契约；模型、normalizer、优化器、
  采样概率、curriculum、学习率和迭代号全部核验通过（此项未增加 PPO 更新）。

## 2026-11-18 T1 落脚事件更新验证

- 相关单元/回归测试：222 passed、8 skipped（可选 GPU 测试）。新增覆盖：左右脚事件求和、
  参考接触不匹配/缺失时仍按物理落脚计分、持续支撑不重复计分、单步掉点防抖、
  reset 基线、终止步快照，以及无效落脚几何不延迟补发。
- [32 环境、2 次更新](../logs/pgmt_stage2_force_isaac_validation/20261118_t1_event_smoke01/training_status.json)：
  完成，源 checkpoint 未改变，历史 curriculum identity 匹配；与上次 T5-only 短测
  保存的完整环境配置逐字段对比，仅 `pgmt_touchdown_quality.settlement` 从
  `support_pattern_mean_v2` 变为 `touchdown_event_v1`，T1 权重确认为 1.5。
- [T1 独立审计](../logs/pgmt_stage2_force_isaac_validation/20261118_t1_event_smoke01/touchdown_event_audit.json)：
  48 个控制步、117 次单脚落脚事件、1742 个持续支撑样本、1 个 reset 行；
  持续支撑均无重复奖励。1053 条独立 PhysX 射线最大高度误差 1.16e-6 m，
  NumPy 方差质量最大误差 5.76e-6；RewardManager 的实际带权奖励全部匹配。
- 同次运行的新版 T5、力/CoP 和四头奖励求和审计均通过；未启动正式长训。

## 2026-11-18 T1/T2 1 N 阈值验证

- 相关单元/回归测试：260 passed、10 skipped（可选 GPU 测试）。新增覆盖严格 1 N 边界、
  T1 竖直支撑与 T2 合力模长的方向区别、非有限力，以及 T3 保持原 10 N 判定。
- [32 环境、1 次 PPO 更新](../logs/pgmt_stage2_force_isaac_validation/20261118_t1t2_1N_smoke01/training_status.json)：
  完成，历史 curriculum identity 匹配，源 checkpoint 未改变。相对上次 T1-event 配置，
  仅将 `pgmt_contact_events.support_threshold_n` 从 10 改为 1，
  新增 `pgmt_reference_contact_threshold_n=1`；奖励权重、T5 配置和其他环境配置不变。
- [实际接触审计](../logs/pgmt_stage2_force_isaac_validation/20261118_t1t2_1N_smoke01/touchdown_event_audit.json)：
  24 个控制步、55 次落脚、870 个持续支撑样本。T2 检出 992 个接触样本，其中 19 个
  处于原 10 N 阈值以下；T3 仍按 10 N 判定。T1/T2 带权奖励与独立计算匹配，
  地形射线最大高度误差 7.75e-7 m，T5 和力/CoP 审计同时通过。
