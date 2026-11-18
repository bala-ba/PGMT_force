# PGMT 力反馈 V1：Stage1 / MuJoCo Warp

V1 将双脚合力、CoP 和 CoP 有效性作为历史状态的一部分，经过 History Attention
后参与 Intent Fusion。当前数据来自仿真接触真值，用于验证力反馈的信息价值。

Isaac Lab 的四头 Stage2 接入、InstantTrack 迁移和 train_03 历史配置恢复现已支持，
见 [Isaac 力反馈 Stage2 训练说明](PGMT_ISAAC_FORCE_STAGE2.zh-CN.md)。下文三头说明针对 Stage1；mjlab Stage2 当前也为四头。

## 接口与网络

默认仍使用 `PGMTActorCritic`，原有命令、96 维 proprio、`10×96` 历史及旧 checkpoint
的元数据、权重键和维度保持兼容。仅指定 `--force-feedback-v1` 时启用
`PGMTForceActorCritic`，其 `proprio_history` 为 `10×108`：

```text
[原 96 维 proprio,
 左脚 Fx, Fy, Fz, CoP_x, CoP_y, valid,
 右脚 Fx, Fy, Fz, CoP_x, CoP_y, valid]
```

`proprio` 仍为 96 维，command window 仍为 `8×61`，critic 原始输入仍为 251 维，
输出仍是 BFS 顺序的 29 维动作及 upper/lower/auxiliary 三个价值头。

每帧历史 embedding 为：

```text
E_history = Linear_proprio(normalize(proprio_96))
          + Linear_force(scale(force_cop_valid_12))

History Attention(query=current proprio, keys/values=history embeddings)
  → Intent Fusion(history token + command tokens)
  → Actor / Critic
```

新增 `Linear_force` 不含 bias，初始化为零；从旧 Stage1 显式迁移后，初始动作和价值
输出与旧模型完全一致，同时该投影可以从 PPO 梯度开始学习。PPO 保存原始 `10×108`
观测并在更新时重新编码，不缓存或 detach 意图 token。

## 接触特征定义

- 只聚合脚部碰撞几何与固定地形的接触，排除自碰撞。支持固定地形子刚体，不能只筛选 body 0。
- 在 GPU 上读取 `mujoco_warp.contact_force` 的逐接触点结果，按 `contact.worldid`
  分环境聚合；力方向统一为环境作用于脚，包含法向力和摩擦力。
- 力和力矩都表达在各自 `ankle_roll_link` 坐标系。力矩为 `Σ (p_i × f_i)`。
  当前脚部 `condim=3`，没有独立接触力偶；若修改 condim，该提取器显式拒绝。
- 参考足底平面为 link 坐标 `z=-0.035 m`。CoP 为
  `x=(z*Fx-My)/Fz`、`y=(Mx+z*Fy)/Fz`。
- 法向力标量之和和足部局部 Fz 均至少为 10 N，且结果有限，CoP 才有效。
  无效 CoP 置零、valid=0；这不清除仍然有效的合力。
- 观测保留 N、m 和 0/1 原始单位。网络内合力除以固定参考重力 327.07660302 N，
  CoP 的 x/y 分别除以 0.15/0.04 m。valid 不参与运行均值方差归一化。
  阈值、尺度和参考平面写入模型元数据及训练配置，严格恢复时校验。
- 这是参考平面上的等效 CoP；倾斜足姿下可能外推到足底轮廓外，不裁剪为假测量。

## 时序与 reset

物理步长 0.005 s、控制步长 0.02 s。提取发生在环境最终 `sim.forward()` 之后的
observation 阶段，因此位置、足部姿态与接触力同一时刻，并用于下一次 action 决策。
训练环中无逐点 Python 循环或接触数据到 CPU 的传输；NaN 检查会检查观测有效性。

重置环境的当前 12 维反馈置零，观测管理器同时重置该环境的历史，不继承上一回合的力。
首次完成物理步进后开始写入新接触特征。连续读取已实现的 observation 不追加历史。
接触/约束缓冲区饱和会触发观测错误，避免使用被截断的合力。

TensorBoard 增加 `ForceFeedback/{left,right}_force_norm_N` 和
`ForceFeedback/{left,right}_cop_valid_fraction`。这些是步进后各环境当前观测的统计，
包含自动重置后的零值，不等同于仅统计支撑期的离线受力均值。

## 运行

使用安装了 `requirements-pgmt-mjlab.txt` 的环境，在仓库根目录运行：

```bash
# 从已有 Stage1 权重初始化；新分支为零，优化器和采样器为新状态。
python scripts/train_pgmt_stage1_mjlab.py --force-feedback-v1 \
  --initialize-from logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt \
  --device cuda:0 --num-envs 128 --max-iterations 100 \
  --num-steps-per-env 24 --save-interval 50 --learning-rate 1e-5 \
  --log-dir logs/pgmt_stage1_force_mjlab/new_run

# 从头训练：不传 --initialize-from。旧模型：不传 --force-feedback-v1。
# 新模型严格恢复：保留相同环境数量/动作库/配置，换新日志目录。
python scripts/train_pgmt_stage1_mjlab.py --force-feedback-v1 \
  --resume logs/pgmt_stage1_force_mjlab/new_run/model_100.pt \
  --device cuda:0 --num-envs 128 --max-iterations 100 \
  --num-steps-per-env 24 --save-interval 50 --learning-rate 1e-5 \
  --log-dir logs/pgmt_stage1_force_mjlab/resumed_run
```

测试使用本机 `/home/lzp/miniconda3/envs/mimiclite-mjlab/bin/python`，并设置
`PYTHONPATH=/tmp/pgmt-mjlab-deps:$PWD` 以加载 MuJoCo 3.8.0 / MuJoCo Warp 3.8.1。
临时依赖目录被清理后，应在独立 Python 环境中安装上述 requirements。

`--learning-rate` 不指定时保留原任务默认值；从预训练 checkpoint 后训练时，上例
显式使用 `1e-5`，避免以从头训练的初始步长大幅改动已有策略。严格 resume 时需要
沿用原运行的初始学习率配置，当前自适应学习率由 checkpoint 恢复。

`scripts/view_pgmt_mjwarp_contacts.py --checkpoint NEW_CHECKPOINT` 根据 checkpoint
元数据自动选择旧模型或力反馈模型，原有 viewer 参数保持不变。

2026-11-18 起 mjlab Stage2 使用六项地形接触奖励和四头 PPO，T1 为 touchdown_event_v1、
T5 为 geometry_v3，支持
InstantTrack Force/CoP checkpoint 的 weights-only 迁移；critic 主干及旧三头加载，
新增 terrain 行随机初始化，见 [Stage2 使用与配置差异](PGMT_MJLAB.zh-CN.md)。实机估计器尚未接入；
短程训练通过只能证明数据、梯度、保存和恢复链路可用，运动控制收益需要同预算的旧模型对照与多动作评估。

## 验证

```bash
python -m pytest tests/test_pgmt_force_feedback.py -q
PMT_FORCE_INTEGRATION=1 PMT_FORCE_TEST_DEVICE=cuda:0 \
  python -m pytest tests/test_pgmt_force_feedback.py -q -k real_feedback
```

覆盖旧 checkpoint 严格兼容、迁移初始输出完全一致、新分支梯度及输入敏感性、无效
CoP 屏蔽、接触对力方向、多环境隔离、足部旋转、排除自碰撞、缓冲区饱和、真实 GPU
逐点力矩与聚合结果比较、历史更新与单环境重置。

## 本次测试记录（2026-11-16）

- 从 `logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt` 迁移，原文件校验和未改变。
- CUDA 0、128 个环境、每次采集 24 步、100 次 PPO 更新，共 307,200 个 transition；
  使用完整 40 个动作文件、136 个动作片段，初始学习率 `1e-5`。
- 完成权重：
  [`model_100.pt`](../logs/pgmt_stage1_force_mjlab/v1_posttrain_128env_100iter_lr1e5_20261116/model_100.pt)。
  详细数值：
  [`validation.json`](../logs/pgmt_stage1_force_mjlab/v1_posttrain_128env_100iter_lr1e5_20261116/validation.json)。
- 新增力反馈投影从全零更新到 L2 范数 `0.055816`，12 个输入列均有更新；
  模型参数与记录的损失均有限。PPO KL 首次 `0.03528`、最后 `0.02445`。
- 严格恢复后继续 2 次更新，保存 `v1_posttrain_resume_verify_20261116/model_102.pt`。
  新模型完成 10 秒、500 帧 headless 接触回放，首个动作以 `motion_finished` 正常结束。
- 新增测试 6 项全部通过（含真实 GPU 聚合及自动重置）；相关原有测试 134 项通过、
  70 项跳过。跳过项未计为通过。

此前默认学习率 `1e-3` 的诊断运行也完成 100 次更新，但首次 PPO KL 为 `13.31`，
因此以上记录使用降低初始学习率后的独立运行；旧训练默认学习率未修改。
本次是 Stage1 的功能与训练连通性测试，尚未进行复杂地形上的同预算控制效果对照。
