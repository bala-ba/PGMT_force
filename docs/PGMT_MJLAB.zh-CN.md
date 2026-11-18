# PGMT Stage1 / Stage2：MuJoCo Warp 训练

新入口使用现有 `mjlab` 后端，不启动 Isaac Sim，不需要 `isaaclab`、`omni` 或 USD。
原有 `scripts/train_pgmt_p1.py`、`scripts/train_pgmt_stage2.py` 及其 shell 启动脚本保持原样。
通用 builder 的默认后端仍是 Isaac；新入口显式选择 mjlab。

## 安装

使用独立的 Python 3.12 环境，在仓库根目录执行：

```bash
python -m pip install -r requirements-pgmt-mjlab.txt
python -m pip install -e .
```

生产训练需要可用的 NVIDIA CUDA 驱动和 CUDA 版 PyTorch。`--device cpu` 仅供小规模
MuJoCo Warp 冒烟验证。不要将 Isaac 的 `requirements-pmt.txt` 安装到此环境。
v8 及转换后的 Isaac v6 `source.mjb` 均由 MuJoCo **3.8.0** 导出，二进制模型依赖版本；新后端会校验
模型和网格哈希，不会在版本不匹配时替换成平地。

接口依据：[mjlab 环境 API](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/envs/manager_based_rl_env.py)、
[MuJoCo Warp 文档](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)。本项目依赖版本见上述 requirements。

## 训练

```bash
# Stage1：默认使用 assets/motions/pgmt_lafan1_50hz 全部动作
python scripts/train_pgmt_stage1_mjlab.py \
  --device cuda:0 --num-envs 1024 --max-iterations 20000 \
  --log-dir logs/pgmt_stage1_mjlab/run1

# Stage2：使用 Isaac v6 布局/P-BFM 对齐难度及 pgmt_clipped_50hz_tcrs_v1 动作库
python scripts/train_pgmt_stage2_mjlab.py \
  --initialize-from logs/pgmt_stage1_mjlab/run1/model_20000.pt \
  --device cuda:0 --num-envs 256 --max-iterations 10000 \
  --log-dir logs/pgmt_stage2_mjlab/run1

# 同阶段恢复：迭代数表示追加更新次数；使用新日志目录
python scripts/train_pgmt_stage2_mjlab.py \
  --resume logs/pgmt_stage2_mjlab/run1/model_10000.pt \
  --device cuda:0 --num-envs 256 --max-iterations 5000 \
  --log-dir logs/pgmt_stage2_mjlab/run2
```

也可以使用公共入口 `scripts/train_pgmt_mjlab.py --stage 1|2`。同时支持
`--num_envs`、`--max_iterations` 等下划线拼写。当前入口为单进程、无窗口训练；
传入 `--headless` 兼容原有习惯。尚未接入 torchrun 多卡训练。

### InstantTrack Force/CoP Stage1 → PMT Stage2 PPO

`--initialize-from` 现在可识别 `instanttrack.pgmt_force_cop.v1`，自动启用
`10×108` 历史和真实接触力/CoP 读取；Stage2 使用 Terrain-Glimpse、六项地形接触奖励和四头 PPO。
默认导入 `ema`（与 InstantTrack 默认评估一致），可用 `--instanttrack-weights model`
选择实时权重。保留 actor、两级注意力、Force/CoP 投影和 actor/command 归一化统计。
critic 主干和 critic 归一化统计也从同一份 checkpoint 权重加载。源输出行 0/1/2
复制到目标 upper/lower/auxiliary 的行 0/1/3，仅新增 terrain 行 2 保持随机初始化。
InstantTrack 的源价值组是 body_tracking/root_tracking/regularization；这属于按行迁移的
权重热启动，并不表示其旧价值目标与 PMT 的 upper/lower/auxiliary 等价。
优化器和课程重新建立，不导入世界模型。PPO 初始探索标准差取
源 checkpoint 的 `action_noise`（本次源为 0.1）；源文件中的 `std=1` 是 BPTT 未使用的兼容参数。
这是跨算法权重初始化，不能使用 `--resume` 加载 InstantTrack 文件。

```bash
python scripts/train_pgmt_stage2_mjlab.py \
  --initialize-from /home/lzp/projects/instantTrack/logs/g1_force_cop_20261116_135357/ckpt_025500.pt \
  --terrain-asset assets/terrain/pgmt_stage2_v6_v8difficulty_mjwarp \
  --learning-rate 1e-6 --device cuda:0 --num-envs 256 --max-iterations 10000 \
  --log-dir logs/pgmt_stage2_force_mjlab/run1
```

PMT 自身的 `pgmt_stage1_force_v1` 也可迁移到 Stage2，保留已有 critic 并随机初始化 terrain 行。
后续 `--resume` 自动识别 `pgmt_stage2_force_v1`，无需再次传 `--force-feedback-v1`。
严格恢复必须沿用原运行的初始配置（包括 `--learning-rate 1e-6`）；当前自适应 LR 从文件恢复。
新地形分支采用原有 PMT 初始化；关闭该分支时，迁移 actor 与源策略应输出一致。

2026-11-18 起，mjlab Stage2 默认使用以下六项地形奖励，共 27 项，
价值组固定为 `[upper, lower, terrain, auxiliary]`。T1 已按用户提供的 Isaac
`--reference-run logs/pgmt_stage2_center2x2/train_03 --touchdown-event` 改为
`touchdown_event_v1`；T5 使用 `geometry_v3`。权重以 Isaac 的
`configs/reward/pgmt_stage2_terrain.yaml` 为基础，将 T1 覆盖为 +1.5；计算复用共享内核。

| 奖励 | 权重 | 计算依据 |
|---|---:|---|
| terrain_touchdown_quality | +1.5 | 每次实际触地结算 `exp(-9 点高度总体方差 / 0.03²)`，两脚质量求和 |
| terrain_reference_contact_match | +1.5 | 实际接触与动作 NPZ 的有效左右脚接触标签匹配率 |
| terrain_slip | -1 | 接触脚 link 原点的世界 XY 速度平方和 |
| terrain_stumble | -20 | 水平法向合力超过 `5×abs(Fz)+10 N` 的脚数 |
| terrain_contact_switching | -0.05 | 短于 0.1 s 的支撑状态切换；参考事件 ±0.04 s 内匹配可豁免一次 |
| terrain_contact_force | -1e-6 | 两脚全物体接触法向合力的模平方和 |

T1 使用最后一个求解子步的地形法向合力世界 Z 分量：`Fz > 1 N` 为支撑，
连续至少 2 个有效非支撑控制步之后进入支撑才触发。初始落地状态只建立基线，
持续站立不重复奖励；一次采样缺失/未知力或 reset 不制造触地事件。事件不受参考接触标签约束。
触地脚按实际 yaw 在长 0.20 m、宽 0.10 m、局部 X 偏置 +0.039 m 的矩形内取 3×3 点。
9 点均须处于有效地形及 torso/yaw 对齐的 2×2 m 扫描覆盖内，否则该脚质量为零。
无触地时不查询 T1 地形；单脚在平地触地得到 `1.5×0.02=0.03`，双脚同落为 0.06。

T5 继续使用四个物理子步地形力的世界 Z 均值和 1/0.5 N 迟滞阈值。
T1 和 T2 的阈值按用户要求统一为 1 N：T1 判定地形合力的世界 Z 分量，
T2 判定全物体接触合力模（T3 复用 T2 的接触状态）。严格大于 1 N 才成立，等于 1 N 不成立。
`train_03` 历史配置使用 10 N；这里保留其 T1 事件/几何定义，并显式覆盖阈值。
奖励在最后一个求解子步后、自动 reset 和参考帧推进前冻结；Actor Force/CoP
观测仍在最终 forward 后采样。MuJoCo 与 PhysX 的求解接触点和法向力数值不保证相同。
T5 内核将事件代价除以 dt，RewardManager 统一乘一次 dt，其余五项直接返回 rate。
动作接触标签缺失、左右脚命名错误或标签形状错误会在初始化时拒绝训练。

Stage2 同时默认启用 **Terrain-Aware Tracking Relaxation**，复用 Isaac 的共享实现。
`configs/task/pgmt_stage2.yaml` 中 `mjlab_tracking_relaxation.enabled: true` 是后端覆盖项；
容差参数继承同文件的 `tracking_relaxation`，最终生效值记录在运行 `config.json` 的
`environment.pgmt_tracking_relaxation` 中。原 Isaac P2 的开关独立保留。
当前 Stage2 启动命令无需增加参数；重新启动后生效，已有训练进程不会热更新。

仅下列三项奖励使用 `max(原始误差 - alpha × 上限 × clamp(L/9,0,1), 0)`，
再按原公式平方、均值和指数计算；默认每个元素的 alpha=1：

| 奖励 | L9 容差上限 |
|---|---:|
| lower_body_position | 0.20 m |
| lower_body_orientation | 0.70 rad |
| lower_joint_position | 0.50 rad |

仅 stairs、slope、boxes 启用；flat、rough 和 L0 容差为零。L 来自每个控制步
机器人实际 torso XY 所在区块的等级，跨区块立即切换；不使用出生课程等级或
manifest 的连续 difficulty。此查询只读区块元数据，不增加地形高度射线。
自动 reset 前冻结本步的误差、容差和参考帧，PPO 收到的分组奖励使用同一快照。
TensorBoard 的 `TrackingRelaxation/` 记录等级缩放、适用环境占比及三项平均容差/奖励增量。

Global Position Correction 仍使用共享 `corrected_command`：参考 anchor 坐标系中的
XY 误差乘 0.5 和速度门控 `clamp(参考 XY 速度/0.5,0,1)`，再将每轴修正限幅 ±0.5 m/s；
用于命令窗口、critic 输入和根部线速度奖励。静止参考的修正为零。

旧三头 Stage2 的训练状态不能按四头配置严格 resume，避免错误恢复优化器和价值目标。
T1 模式、几何参数和支撑阈值已写入运行/检查点契约；旧持续风险 T1 的 checkpoint
不能按当前事件奖励配置严格 resume，应从 Stage1 初始化新训练。
跟踪放宽的开关、容差、元素名称及地形映射规则也写入契约；旧关闭放宽的 Stage2
checkpoint 不兼容当前默认配置的严格 resume。使用前述 `--initialize-from` 命令启动新训练。

历史持续风险 T1 的验收记录：`logs/stage2_geometry_v3_acceptance_20261118/acceptance.json`。
实际使用指定 `ckpt_025500.pt` 的 EMA，迁移参数逐张量精确比对通过；新增 terrain
输出行保持迁移前随机值。v6 五类地形、10 个环境、交替 L0/L9，8 步奖励检查含一次
强制自动 reset，六项 rate 有限、每步 4 次物理采样、分组和与标量奖励一致。
随后以单段真实 walking 动作完成 EMD 评估及 1 次 PPO 更新（240 条训练样本），
四个 critic 输出行均更新，actor/critic/Force/CoP/Terrain-Glimpse 的 Adam 状态有限且非零，
`model_1.pt` 保存及严格恢复通过。另独立检查全部 862 段、430138 帧的接触标签。
这是链路验收，未验证长期收敛；现有地形 BVH 查询瓶颈仍使此次 PPO 采集耗时较高。

事件 T1 的历史 GPU 验收（当时 T1 为 10 N、T2 为 1 N）：
`logs/stage2_t1event_t5v3_acceptance_20261118/acceptance.json`。
使用 cuda:1、seed=722、v6 五类地形的 10 个环境，24 步中检测到 12 次触地，
独立 NumPy 事件判定及方差奖励一致；108 个采样高度与原生 MuJoCo 射线核对通过，
最大高度误差约 1.07e-6 m。强制 reset、四组奖励求和、4 子步采样检查通过；
随后完成 1 次 PPO 更新（240 条训练样本），四个 critic 输出均更新，保存和严格恢复通过。
随后将 T1/T2 统一为 1 N；当前后端测试覆盖 0.75/1.0/1.01 N 的事件及接触匹配边界。

当前跟踪放宽及 T1/T2=1 N 的 GPU 验收：
`logs/stage2_relaxation_acceptance_20261118/acceptance.json`（同目录有 `config.json`、`console.log`）。
使用指定 Stage1 checkpoint 的 EMA、cuda:1、seed=722，v6 五类地形共 10 个环境，
交替 L0/L9。24 步中逐步独立核对实际 torso 区块、三项容差/残差/奖励、自动 reset
前的参考帧和四组奖励求和；L9 三项上限均生效。12 次触地、108 条高度射线核对通过，
最大高度误差约 8.34e-7 m。完成 1 次 PPO 更新（240 条训练样本），四个 critic
输出行均更新，各分支 Adam 状态有限且非零，保存和严格恢复通过；修改放宽配置的
检查点状态被拒绝。相关 CPU 回归测试 52 项通过，另 1 项需显式开启的仿真集成测试跳过、
2 项原生仿真测试未纳入该测试命令；上述真实 GPU 验收独立执行。此次未验证长期收敛或 8192 环境规模。

运行当前 T1 事件奖励及跟踪放宽的有限验收（24 步事件/几何/容差核对、1 次 PPO 更新和恢复）：

```bash
python scripts/validate_pgmt_mjlab_terrain_rewards.py \
  --checkpoint /home/lzp/projects/instantTrack/logs/g1_force_cop_20261116_135357/ckpt_025500.pt \
  --output logs/stage2_relaxation_acceptance_new --device cuda:1 --seed 722
```

本次 32 环境对照中，任务默认 LR=1e-4 的首轮 approximate KL 约 87.4；显式使用
1e-6 后约为 .031，后续约 .02。因此上述迁移示例使用较小起始 LR，原任务默认值未改。
自适应 LR 仍可能降至 1e-7；短训不能证明长期学习速度或收敛，不能直接沿用默认步长作无监控长训。

### 与 Isaac 配置的边界（更新至 2026-11-18）

两套实现的配置并非完全一致。以下比较指仓库中的默认配置；实际运行覆盖项应以日志为准。

| 项目 | MuJoCo Warp Stage2 | Isaac Stage2 |
| --- | --- | --- |
| 控制率 / 仿真步长 | 50 Hz / 0.005 s，decimation=4 | 相同 |
| PPO | 24 步、5 epoch、4 minibatch、LR=1e-4、gamma=.99、lambda=.95、clip=.2、entropy=.005 | 三头 P2 相同 |
| 地形 / 动作 | `pgmt_stage2_v6_v8difficulty_mjwarp`、5 类、10 级、862 段完整动作、EMD | 对应 `pgmt_stage2_v6_v8difficulty` 归档；共享动作采样实现。Isaac 全局默认配置仍指向 v8，实际选择须以运行配置为准 |
| 地形输入 | 21×21、2×2 m、10% dropout、±.06 m 噪声、±3 m 裁剪 | 相同共享实现 |
| 本体观测噪声 | 姿态 ±.05、角速度 ±.2、关节角 ±.01、缩放后的关节速度 ±.025 | 相同 |
| torso COM | XYZ ±(.025,.05,.05) m，startup | 相同范围 |
| 摩擦 | 脚部 geom 共享 U(.3,1.2)，脚优先级 1 | 全身 static U(.3,1.6)、dynamic U(.3,1.2)，64 桶 |
| 恢复系数 | 没有同等随机化 | U(0,.5) |
| 关节零位偏置 | 无 | startup ±.01 rad |
| 推扰 | 每 1–3 s；XY ±.5、Z ±.2 m/s；roll/pitch ±.52、yaw ±.78 rad/s | 间隔相同；XYZ ±.2；roll/pitch ±.22、yaw ±.3 |
| 初始课程 | 全部 L0 | 随机 L0/L1 |
| 时间上限 | 500 控制步 | 同为 10 s，另有完成动作距上限 2 步的边界处理 |
| 奖励 / 价值头 | 27 项 / 4 头，T1 事件 + T5 geometry_v3，默认启用三项下肢 tracking relaxation | T1 对齐上述命令的 touchdown-event；其他项共用纯张量内核；Isaac P3 启用相同 tracking relaxation |

Force/CoP 历史是此次显式扩展，Isaac 默认 P2 仍为 `10×96`。
以上相同项表示参数/共享公式对齐，不表示 MuJoCo 与 PhysX 接触求解结果相同。
本次未改动原有域随机化分布；完整配置记录在训练目录的 `config.json`。

### 实际地形与观测图

2026-11-17 地形纠正：本次使用用户指定的 **v6 布局 + P-BFM 对齐难度**，
不是七类 v8。源为 `assets/terrain/pgmt_stage2_v6_v8difficulty`，保留
flat / stairs / slope / boxes / rough 五类、10 级、24×24 m 区域、
每区 4×4 个 6×6 m 小块，以及中心 2×2 出生小块（编号 5/6/9/10）。
粗糙地形保留 Isaac 的均匀噪声生成结果；方块保留原网格，没有替换为 v8 的 Perlin/离散方块。

MuJoCo 专用资产从已完成的 Isaac PLY 读取，不重新采样难度或生成地形：
轴对齐盒体逐组件转换；高度场保留原高度和三角形对角线。
开放平面/高度场在最低表面以下增加实体底面；不在坑内覆盖零高度平面。
新后端的默认地形通过 `configs/task/pgmt_stage2.yaml` 的 `mjlab_terrain_asset` 指定。
首次准备环境时可执行以下导出（拒绝覆盖已存在目录）：

```bash
python scripts/export_pgmt_isaac_terrain_mjwarp.py
```

800 个小块全部校验，1,825,440 条表面射线及 4,620 个接缝点；
最大高度误差 6.70e-8 m。MuJoCo 3.8 的 CPU `mj_ray` 在精确网格线上
偶发漏交：152 个点以 (3e-8,7e-8) m 偏移重新查询**原网格与原生碰撞体两边**，
再次检查高度一致；未改动模型表面。结果见资产内 `conversion_validation.json`。
这是几何表面一致性检查，不表示 PhysX 与 MuJoCo 接触动力学相同。

```bash
MUJOCO_GL=egl python scripts/render_pgmt_mjlab_terrain.py --output logs/terrain_preview
python scripts/audit_pgmt_stage2_mjlab.py --checkpoint STAGE1_OR_STAGE2.pt \
  --output logs/terrain_audit --device cuda:0 --steps 100
```

渲染器直接调用训练的 `add_native_terrain`，保留原生 boxes/heightfields 并检查源模型一致性。
分族图只改变显示颜色、光照和可见性。rollout 审核保存实际增强后的高度图、有效掩码、
Force/CoP、有效接触摩擦及缓冲容量；不将短 rollout 解释为地形通过率或收敛验证。

常用参数：

- `--motion-files FILE_OR_DIRECTORY ...`：加载所有指定 NPZ，递归扫描目录，不只选择第一个文件。
- `--terrain-asset DIR`、`--compatibility-file CSV`：覆盖 Stage2 地形和动作兼容表。
- `--num-steps-per-env 24`、`--save-interval 200`、`--seed 42`。
- `--max-runtime-s SECONDS`、`--stop-file PATH`：在完整 PPO 更新结束后保存并退出；SIGINT/SIGTERM 同样处理。
- `--max-iterations 0`：初始化或恢复后保存，不执行 PPO/EMD 更新。
- `--dry-run`：校验配置、全部动作格式、Stage2 兼容表和原生地形编译，不启动物理仿真。

```bash
python scripts/train_pgmt_stage1_mjlab.py --dry-run
python scripts/train_pgmt_stage2_mjlab.py --dry-run
```

## 训练语义

策略、动作、参考动作及 critic 输入均保留 PMT 的 BFS 关节/刚体顺序；mjlab 的
MJCF 顺序只出现在物理状态读写边界。无需预先运行 `pmt_npz_to_mjlab.py`；没有
名字字段的 NPZ 必须是 PMT 原始 BFS 格式，有名字字段的 NPZ 按名字重排。
检查每段的维度、有限值、wxyz 单位四元数，以及帧率与控制频率的一致性。

复用 PGMT 的 `96` 维 proprio、`10×96` 历史、`8×61` 未来命令、`251` 维 critic、
Stage1 为 21 项奖励和 upper/lower/auxiliary 三个价值头；Stage2 为 27 项、四个价值头。
分组奖励从 reset 前冻结的 reward
rate 计算，仅乘一次控制 dt；终止和截断同时发生时，失败优先，不进行 timeout bootstrap。
原始动作按 `default_joint_pos + scale * action` 转成位置目标。

Stage1 使用 motion → temporal bin → frame 的自适应分箱采样，保留 EMA 和起始 bin
归因。为保证至少有一个有效转移，不从动作最后一帧开始。动作完成通过环境截断
显式 reset，并清空历史。物理模型、接触、随机化由 mjlab 提供，因此新后端训练不等同于
Isaac 轨迹的逐步复现。

Stage2 通过 **`pgmt_stage2` 的 mjlab 入口**启用四头地形训练（Isaac P2 配置保持原状）：

- 加载已审核的原生碰撞 boxes/heightfields，保留全部高度样本及世界位姿；当前默认为上述 v6 转换资产。
- 复用已有兼容表、地形分配和进阶课程；重置时按实际 mjlab 碰撞几何生成保守球覆盖，
  对参考姿态作整体旋转、平移和抬升，检查初始姿态及未来参考的边界。
- 使用同一地形的审核网格做 Warp 高度/距离查询，生成实际 torso yaw 下的 `21×21`
  地形高度图及有效掩码，保留 dropout、±6 cm 噪声和 ±3 m 裁剪。
- 保留 0.2 秒连续失败计时、1 m 根部 XY 阈值、0.4 m torso/手腕高度阈值；
  只对手腕使用末端加速度奖励。
- 开始训练及配置间隔处，在独立的 mjlab 平地环境中跑完整动作库、计算 EMD 并更新权重。
  评估不修改训练世界或归一化统计，评估后恢复训练 RNG。首次训练日志会先显示 EMD 进度。

四头 P3 任务 `pgmt_stage2_terrain` 的接触专项奖励不属于本次两阶段入口，仍通过原有
Isaac 入口训练；builder 对未迁移任务显式报错。

## 检查点与验证

`--initialize-from` 支持已有 Stage1（Isaac 或 mjlab）到 Stage2 的权重迁移，也支持
同策略的显式 weights-only 初始化。跨仿真器的参数初始化需要后续训练适应物理差异。
`--resume` 仅接受本后端同配置检查点，恢复 Adam、学习率、迭代数、RNG、采样器和
地形课程；物理 episode 重新开始。更换动作内容、地形、环境数量、控制率或后端版本
会拒绝严格恢复。源码变更后应重新验证兼容性。

每次运行生成 `config.json`、`training_status.json`、TensorBoard 日志和 `model_N.pt`，
其中 `N` 表示已完成更新次数。拒绝覆盖已有训练目录。

```bash
# 单元测试与实际 CPU MuJoCo Warp rollout/自动 reset/奖励积分检查
PMT_MJLAB_INTEGRATION=1 python -m pytest tests/test_pgmt_mjlab.py -q
```

早期后端验收包含 CPU 与 RTX 5090 CUDA 后端的两阶段 PPO 更新、Stage1→Stage2 初始化、
独立 EMD 评估，以及 CPU 后端的两个阶段检查点恢复。使用两环境、四步 rollout 和短动作
片段验证训练链路；新增与相关回归测试合计 129 项通过、1 项跳过。
大规模 CUDA 吞吐量、长时间训练收敛和 Isaac/MuJoCo 数值一致性不由这些冒烟检查保证。

2026-11-17 的**早期 v8** Force Stage2 验证位于 `logs/stage2_force_audit_20261117`：RTX 5090、
32 环境、24 步 rollout、862 段完整动作 EMD、10 次 PPO 更新（7680 transitions）；
默认 LR 试验另完成 10→12 次严格恢复。actor、critic、Force/CoP、Terrain-Glimpse
均有有限非零 Adam 更新。新增及相关回归 146 项通过、7 项跳过。
`training_validation_lr1e6.json` 是较小起始 LR 的最终验证，`audit_report.json` 汇总配置差异。
该早期目录的地形图和短训使用了错误的七类 v8，不是用户指定的 v6；
不能作为 v6 地形的运行验收。纠正后的图像与运行日志独立存放于
`logs/stage2_force_v6_audit_20261117`。

纠正后的 v6 实测（2026-11-17 至 18）：10 环境覆盖五类地形的 L0/L9，
完成 100 控制步，观测/动作/奖励均有限，接触峰值 200/1280、约束峰值 128/512；
奖励分组加和最大误差 5.96e-8。32 环境完成 862 段完整动作 EMD，以及
1 次完整 PPO 更新（24 步、768 transitions、20 个 minibatch 更新），保存
`train/model_1.pt`。检测到低吞吐后，在当前更新结束时正常保存并退出，未完成原计划的 10 次。
actor、critic、Force 投影及地形分支都有非零 Adam 状态；这是可训练性验证，不是收敛验证。

这份完整 v6 原生模型含 14,404 个碰撞体，当前短测采样 120.17 s、PPO 计算 2.35 s，
约 6.3 samples/s。默认与 NXN 碰撞搜索对照分别约 3.69/3.67 s 每控制步，未见明显改善，
因此未更改训练默认碰撞算法。GPU 同时有既有训练和对照测试，这不是独占吞吐基准；
初始化调用栈显示容量估算较慢，每步瓶颈还需专项 profiling。当前不宜直接启动长训。
上述域随机化差异继续存在；本次纠正地形没有改动其分布。

2026-11-18 吞吐定位补充：`logs/stage2_v6_profile_20261118/diagnosis.json`
记录了同步阶段计时、CUDA trace 和同进程对照。32 环境每个控制步两次地形高度查询
合计约 2.09 s（躯干高度终止判断 32 条射线，以及地形观测 14,112 条射线）；
4 次物理子步加 forward 约 31.5 ms，策略约 2.7 ms，Force/CoP 约 1.7 ms。
因此稳态主瓶颈是查询 BVH，不能归因于碰撞搜索或 PPO 更新。
保持 GPU 顶点、三角形、实际射线坐标及核函数完全相同，仅在诊断进程中重建
同算法 LBVH：单次查询从约 1.1 s 降至约 2.4 ms，高度差为 0；后续 3 个完整
控制步平均从 2.077 s 降至 .0789 s（约 15.4→405 samples/s）。该短测仍共享 GPU，
不能视为长期 PPO 吞吐保证。重建效果可复现，但初始化后原始 BVH 低效的底层
触发条件尚未确定；独立测试排除了旧核函数缓存和精确网格线坐标两个猜测。
这次只新增诊断脚本与报告，生产环境的 BVH 构建代码尚未修改。
