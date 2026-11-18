通过 `train_pgmt_stage2_native.sh --initialize-from <InstantTrack Force checkpoint>` 新建的
Isaac Stage2 训练使用下文“对比 checkpoint policy”表的 **99 维单帧字段**。
当前输入为 `proprio[111]`（99 维本体加当步双脚足力、CoP 及有效标记 12 维）、
`proprio_history[10,99]` 和独立的 `force_history[10,12]`。
actor 首层为 239 维，critic 首层为 389 维，由当前特权状态 185 维、当步双足反馈 12 维、
干净 21×21 地形图的 MLP 编码 64 维和共享 intent 128 维组成。
下表中的 PMT 251→379 维 critic、96 维 proprio 和 108 维 Force 历史对应原 PGMT 布局。
新训练的字段顺序、参考高度和归一化见[新版 Isaac Force Stage2 critic](PGMT_ISAAC_FORCE_STAGE2.zh-CN.md#critic-的-389-维输入)。

| 记录对象 | 输入定义与核对依据 |
| --- | --- |
| 当前 PMT 四头模型 | 默认 `pgmt_stage2_terrain` / `PGMTActorCritic`；[观测构造](../pmt_tasks/pgmt_math.py)、[网络输入路径](../motion_tracking_rl/networks/pgmt_actor_critic.py)、[任务配置](../configs/task/pgmt_stage2_terrain.yaml) |
| 当前 PMT 可选 Force/CoP 模型 | `PGMTForceActorCritic`；仅启用该变体时，历史单帧由 96 维扩展到 108 维；[网络](../motion_tracking_rl/networks/pgmt_force_actor_critic.py)、[力反馈定义](../motion_tracking_rl/pgmt_force_feedback.py) |
| 对比 checkpoint | `/home/lzp/projects/instantTrack/logs/g1_steady_env4096_20261108/ckpt_004000.pt`；`pi`、`pi_live` 的输入形状均已从权重核对 |
| 对比 checkpoint 的 critic | 指定文件未包含 critic 权重；同目录 `last.pt` 的迭代、保存时间、参数配置及 `pi` 全部张量与指定文件一致，其 critic 首层为 `914 → 256`；字段定义依据与该 run 的 `source_hashes.json` 匹配的归档 `train.py`、`mdp.py`、`networks.py`、`dr.py`、`g1_constants.py` |
| 对比模型归档源码 | `/home/lzp/projects/instantTrack/migration/pmt_stage1/reference/instanttrack/src/instanttrack/`；参考窗口定义见其中 `motion.py` |
| 表格形状与索引约定 | 形状省略 batch 维；索引从 0 开始，`[a:b]` 左闭右开；历史按旧到新排列，包含当前帧；常规控制周期为 20 ms |

| 模型 / 输入层 | 输入项 | 单样本形状 / 维度 | 输入路径 |
| --- | --- | --- | --- |
| PMT policy：外部观测 | 当前 `proprio` | `96` | 直接进入 actor MLP，并作为历史 attention 的 query 来源 |
| PMT policy：外部观测 | `proprio_history` | 默认 `10 × 96`；Force/CoP 变体 `10 × 108` | history attention → IFM |
| PMT policy：外部观测 | `command_window` | `8 × 61` | command embedding → IFM；同时输入 Terrain-Glimpse selector |
| PMT policy：外部观测 | `terrain_map`、`terrain_valid` | 各 `21 × 21`，合计 882 个数值 | Terrain-Glimpse → 4 个地形 token → IFM |
| PMT actor MLP | 当前归一化 `proprio` + 共享 `intent` | `96 + 128 = 224` | 两者直接拼接；Force/CoP 变体仍为 224 维 |
| PMT critic MLP | 当前归一化特权观测 + 同一共享 `intent` | `251 + 128 = 379` | `upper`、`lower`、`terrain`、`auxiliary` 四个输出共享这份输入；没有额外的 terrain-head 专属输入 |
| PMT critic：间接输入 | policy 当前观测、历史、参考窗口、地形图与 mask；可选力反馈历史 | 共同压缩为 `intent[128]` | 经共享 IFM 进入 critic；251 维特权观测本身不包含这些完整窗口或地形图 |
| 对比 checkpoint：policy 外部输入 | `obs_hist` + `goal` | `10 × 99` + `730` | 历史归一化后展平；goal 单独编码 |
| 对比 checkpoint：motion encoder | 归一化 `goal` | `730 = 10 × 73` | 输出 `z[32]`，经 `tanh` |
| 对比 checkpoint：policy decoder | `z` + 归一化历史观测 | `32 + 10 × 99 = 1022` | 先拼接 z，再拼接历史；首层权重形状为 `2048 × 1022` |
| 对比 checkpoint：policy 参考旁路 | 原始 goal 首个未来帧的参考关节角 `q_ref(f+1)` | `29`，来自 `goal[15:44]` | 不新增外部观测；以 `(q_ref − ref_q0) / ref_alpha` 加到 decoder 输出 |
| 对比 checkpoint：critic | 当前特权状态 / 跟踪误差 / 仿真参数 + 原始参考窗口 | `184 + 730 = 914` | 整体经独立 RunningNorm；直接输入 goal，不使用 policy 的 z 或历史编码 |

| PMT 当前 proprio / 每帧历史：字段 | 单帧索引 | 维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 躯干参考姿态误差 | `[0:6]` | 6 | `R_torso_actualᵀ R_torso_ref` 的 6D 表示；按第一列 xyz、第二列 xyz 拼接 |
| 实际根部角速度 | `[6:9]` | 3 | pelvis 根部坐标系中的 xyz 角速度，rad/s |
| 实际关节角 | `[9:38]` | 29 | `q`，绝对关节角，rad；不是 `q − q_default` |
| 实际关节角速度 | `[38:67]` | 29 | `0.05 × qdot`，原单位 rad/s |
| 已下发的 PD 目标关节角 | `[67:96]` | 29 | articulation 保存的 `joint_pos_target`，rad；不是原始归一化 action |
| 单帧合计 | `[0:96]` | 96 | 当前帧与历史前 96 维使用同一个 actor observation normalizer |
| 历史时间窗口 | 帧索引 `[0:10]` | `10 × 96` | `t−9,…,t`，20 ms 间隔；最旧与最新相差 180 ms |
| 训练观测噪声 | 对应上述字段 | 96 | 默认独立均匀噪声半宽依次为姿态 `0.05`、角速度 `0.2`、关节角 `0.01`、已缩放关节角速度 `0.025`、PD 目标 `0`；之后进入 running normalization |

| PMT command_window：每个参考帧字段 | 帧内索引 | 每帧维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 参考关节角 | `[0:29]` | 29 | 对应参考帧的 `q_ref`，rad |
| 参考关节角速度 | `[29:58]` | 29 | `0.05 × qdot_ref` |
| 修正后的参考躯干线速度 | `[58:61]` | 3 | 在该参考帧躯干完整姿态坐标系中表达；xy 加上由当前实际 / 参考躯干位置偏差产生的速度修正；z 强制为 0 |
| 每帧合计 | `[0:61]` | 61 | 共享 61 维 command normalizer，无观测噪声；不是完整参考根部位姿 |
| 参考帧偏移 | `[0,1,3,7,15,31,63,127]` | 8 帧 | 相对当前参考帧；对应 `[0,0.02,0.06,0.14,0.30,0.62,1.26,2.54]` 秒；超出当前片段末尾时截到最后一帧 |
| 整个窗口 | — | `8 × 61 = 488` | 保留逐帧结构，command embedding 将每帧映射到 128 维 |

| PMT 地形与 IFM 输入项 | 形状 / 维度 | 含义、坐标系与输入路径 |
| --- | --- | --- |
| 地形高度图 `terrain_map` | `21 × 21 = 441` | 实际 torso 的 yaw 坐标系；覆盖 `2 m × 2 m`，间隔 0.1 m；行 y、列 x，各从 −1 m 到 +1 m；值为 `sensor_world_z − ground_world_z` |
| 地形有效性 `terrain_valid` | `21 × 21 = 441` | 同一次观测的有效 mask：地图内、命中有限且未 dropout；输入网络时为 0/1 |
| 高度处理 | 441 | 默认 10% dropout、独立均匀 `±0.06 m` 噪声、裁剪至 `[-3,3] m`；网络再除以 3；无效点保留填充值并配合 mask，不置零 |
| 历史 token | `1 × 128` | 当前 proprio embedding 查询 10 帧历史 embedding 的 attention 输出 |
| 地形位置 selector 输入 | `441 + 441 + 128 + 488 = 1498` | 高度图、mask、历史 token、归一化参考窗口按此顺序拼接 |
| 地形 patch encoder 输入 | `4 × 52` | 每个 patch：`5 × 5` 高度 25 维 + 有效 mask 25 维 + 中心 xy 2 维；中心位于 torso-yaw 坐标系，单位 m |
| 地形 token | `4 × 128` | 四个 patch 分别编码；patch 有效性用于 IFM attention mask |
| IFM query | `1 × 128` | 历史 token |
| IFM key / value | `13 × 128` | 按顺序为历史 token 1 个、参考 command token 8 个、地形 token 4 个 |
| IFM 输出 `intent` | 128 | 同时拼接到 actor 与 critic 的最终 MLP 输入 |

| PMT critic：直接特权输入与融合输入 | MLP 输入索引 | 维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 躯干参考姿态误差 | `[0:6]` | 6 | 与 proprio 相同的 torso 相对参考姿态 6D，使用干净状态 |
| 实际根部线速度 | `[6:9]` | 3 | pelvis 根部坐标系，m/s |
| 实际根部角速度 | `[9:12]` | 3 | pelvis 根部坐标系，rad/s |
| 实际关节角 | `[12:41]` | 29 | `q`，rad |
| 实际关节角速度 | `[41:70]` | 29 | `0.05 × qdot` |
| 已下发的 PD 目标关节角 | `[70:99]` | 29 | `joint_pos_target`，rad |
| 当前参考关节角 | `[99:128]` | 29 | `q_ref(t)`，rad |
| 当前参考关节角速度 | `[128:157]` | 29 | `0.05 × qdot_ref(t)` |
| 全部刚体的相对位置误差 | `[157:247]` | `30 × 3 = 90` | 实际各 body 相对实际 torso 的位置转到实际 torso-yaw 系；参考各 body 相对参考 torso 的位置转到参考 torso-yaw 系；两者作“实际 − 参考”，按 body 顺序、每 body xyz 展平，单位 m |
| 当前参考根部高度 | `[247:248]` | 1 | 参考 pelvis 的 world z 减去环境 origin z；不是实际根部高度 |
| 当前修正后的参考躯干线速度 | `[248:251]` | 3 | 参考 torso 坐标系；xy 含位置误差修正，z 为 0 |
| 直接特权观测合计 | `[0:251]` | 251 | 当前帧、无观测噪声，整体使用独立 critic observation normalizer |
| 共享 IFM 输出 | `[251:379]` | 128 | 由 policy 侧的当前 / 历史观测、参考窗口、地形生成；启用 Force/CoP 时也包含其历史编码；不经过 251 维 critic normalizer |
| critic MLP 输入合计 | `[0:379]` | 379 | 四个 value 输出共享；Force/CoP 变体也保持 379 维 |

| PMT 可选 Force/CoP：历史新增字段 | 108 维历史帧内索引 | 维度 | 含义与缩放 |
| --- | --- | --- | --- |
| 原有 proprio | `[0:96]` | 96 | 与默认模型相同 |
| 左脚接触合力 | `[96:99]` | 3 | 左 `ankle_roll_link` 坐标系中的 xyz 力；默认除以 `327.07660302 N` |
| 左脚 CoP | `[99:101]` | 2 | 脚底平面 `z=−0.035 m` 的等效压力中心 xy；分别除以 `0.15 m`、`0.04 m`；无效时这两维置零 |
| 左脚 CoP 有效性 | `[101:102]` | 1 | 0/1；该位无效不会把接触力三维一起清零 |
| 右脚接触合力 | `[102:105]` | 3 | 右 `ankle_roll_link` 坐标系；与左脚相同缩放 |
| 右脚 CoP | `[105:107]` | 2 | 与左脚相同定义与缩放 |
| 右脚 CoP 有效性 | `[107:108]` | 1 | 0/1 |
| 历史合计与融合 | — | `10 × 108` | 前 96 维正常归一化并投影；新增 12 维固定缩放后独立线性投影，两者相加得到每帧 128 维 embedding；当前 proprio 仍是 96 维，critic 直接特权输入仍是 251 维 |

| 对比 checkpoint policy：每帧 obs_hist 字段 | 帧内索引 | 每帧维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 相对默认姿态的关节角 | `[0:29]` | 29 | `q − q_default − joint_offset`，rad；joint_offset 为仿真编码器偏置 |
| 实际关节角速度 | `[29:58]` | 29 | `qdot`，rad/s；实际 policy 路径没有额外乘 0.1 |
| 实际根部角速度 | `[58:61]` | 3 | pelvis 坐标系，rad/s；实际 policy 路径没有额外乘 0.25 |
| 投影重力方向 | `[61:64]` | 3 | `R_pelvis_actualᵀ × (0,0,−1)`，无量纲 |
| 上一步动作 | `[64:93]` | 29 | `a_prev`，归一化动作；不是 PD 目标角 |
| 根部参考姿态误差 | `[93:99]` | 6 | `R_pelvis_actualᵀ R_pelvis_ref` 的前两列，按行展平为 `R00,R01,R10,R11,R20,R21`；与 PMT 的 6D 展平顺序不同 |
| 单帧合计 | `[0:99]` | 99 | 使用 checkpoint 保存的 99 维 `obs_norm`，`(x−mean)/sqrt(var+0.01²)` |
| 历史窗口 | 帧索引 `[0:10]` | `10 × 99 = 990` | `t−9,…,t`，20 ms 间隔，最新帧在末尾；先逐帧归一化，再按时间 / 特征展平 |
| 训练观测噪声 | 对应上述字段 | 99 | 独立均匀噪声半宽依次为关节角 `0.01`、关节速度 `0.5`、角速度 `0.2`、重力 `0.05`、上一步动作 `0`、参考姿态误差 `0.05`；之后进入 obs_norm |

| 对比 checkpoint：policy encoder 与 critic 共用的原始 goal 字段 | 每个未来帧内索引 | 每帧维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 未来参考根部位移 | `[0:3]` | 3 | `H_ref(t)ᵀ × (p_ref(t+k)−p_ref(t))`；当前参考 pelvis 的 yaw 坐标系，m |
| 未来参考根部姿态 | `[3:9]` | 6 | `H_ref(t)ᵀ R_ref(t+k)` 的 6D 表示；前两列按行展平；H 只含当前参考 yaw |
| 未来参考根部线速度 | `[9:12]` | 3 | 该未来参考帧 pelvis 完整姿态坐标系，m/s；保留 xyz |
| 未来参考根部角速度 | `[12:15]` | 3 | 该未来参考帧 pelvis 完整姿态坐标系，rad/s |
| 未来参考关节角 | `[15:44]` | 29 | `q_ref(t+k)`，rad |
| 未来参考关节角速度 | `[44:73]` | 29 | `0.1 × qdot_ref(t+k)` |
| 每帧合计 | `[0:73]` | 73 | 仅由参考动作生成，不包含实际机器人状态 |
| 未来帧偏移 | `[1,2,3,4,5,6,7,8,9,10]` | 10 帧 | 对应未来 `0.02,…,0.20 s`，不包含当前参考帧；片段末尾截断到最后一帧 |
| 整个 goal | `[0:730]` | 730 | 按帧、再按上述字段顺序展平；policy 使用独立 `goal_norm`；critic 将原始 goal 拼到其输入末尾后统一归一化 |

| 对比 checkpoint critic：字段 | critic 输入索引 | 维度 | 含义、坐标系与处理 |
| --- | --- | --- | --- |
| 实际根部高度 | `[0:1]` | 1 | pelvis 的 world z，m |
| 实际根部倾斜姿态 | `[1:7]` | 6 | `H_actualᵀ R_actual`，去除实际 yaw 后的姿态；前两列按行展平 |
| 实际根部线速度 | `[7:10]` | 3 | pelvis 完整姿态坐标系，m/s |
| 实际根部角速度 | `[10:13]` | 3 | pelvis 完整姿态坐标系，rad/s |
| 实际关节角 | `[13:42]` | 29 | `q`，rad；不是 policy 的偏置扣除后关节角 |
| 实际关节角速度 | `[42:71]` | 29 | `qdot`，rad/s；无额外固定倍率 |
| 当前参考关节角误差 | `[71:100]` | 29 | `q_ref(t) − q`，rad |
| 当前参考刚体位置误差 | `[100:142]` | `14 × 3 = 42` | 将参考身体按实际 pelvis 的 xy 与相对 yaw 对齐，保留参考 pelvis 高度，再计算“对齐后参考 − 实际”，转到实际 pelvis-yaw 坐标系；按 tracked-body 顺序、每 body xyz 展平 |
| 当前参考根部位置误差 | `[142:145]` | 3 | `H_actualᵀ × (p_ref(t)−p_actual)`，m；包含根部全局漂移误差 |
| 当前参考根部姿态误差 | `[145:151]` | 6 | `R_actualᵀ R_ref(t)`，前两列按行展平 |
| 仿真摩擦系数 | `[151:152]` | 1 | xi 的 friction；先按 `(friction−0.75)/0.45` 固定归一化 |
| 仿真关节编码器偏置 | `[152:181]` | 29 | xi 的 joint_offset；默认各关节除以 `0.01 rad` |
| 仿真躯干质心偏移 | `[181:184]` | 3 | xi 的 torso CoM xyz 偏移；各轴除以 `0.05 m` |
| 当前特权输入合计 | `[0:184]` | 184 | 干净状态 / 跟踪误差 151 维 + 已固定归一化的仿真参数 33 维 |
| 原始未来参考窗口 | `[184:914]` | 730 | 上表的完整 10 帧 goal；不经过 policy encoder，也不使用 policy 的 goal_norm |
| critic 输入合计 | `[0:914]` | 914 | 整体经过 critic 自己的 RunningNorm；没有历史观测、上一步动作、地形图、足部接触力或 CoP 输入 |

| 关节输入向量的元素索引 | 当前 PMT：BFS 顺序 | 对比 checkpoint：G1 SDK 顺序 |
| --- | --- | --- |
| 0 | `left_hip_pitch_joint` | `left_hip_pitch_joint` |
| 1 | `right_hip_pitch_joint` | `left_hip_roll_joint` |
| 2 | `waist_yaw_joint` | `left_hip_yaw_joint` |
| 3 | `left_hip_roll_joint` | `left_knee_joint` |
| 4 | `right_hip_roll_joint` | `left_ankle_pitch_joint` |
| 5 | `waist_roll_joint` | `left_ankle_roll_joint` |
| 6 | `left_hip_yaw_joint` | `right_hip_pitch_joint` |
| 7 | `right_hip_yaw_joint` | `right_hip_roll_joint` |
| 8 | `waist_pitch_joint` | `right_hip_yaw_joint` |
| 9 | `left_knee_joint` | `right_knee_joint` |
| 10 | `right_knee_joint` | `right_ankle_pitch_joint` |
| 11 | `left_shoulder_pitch_joint` | `right_ankle_roll_joint` |
| 12 | `right_shoulder_pitch_joint` | `waist_yaw_joint` |
| 13 | `left_ankle_pitch_joint` | `waist_roll_joint` |
| 14 | `right_ankle_pitch_joint` | `waist_pitch_joint` |
| 15 | `left_shoulder_roll_joint` | `left_shoulder_pitch_joint` |
| 16 | `right_shoulder_roll_joint` | `left_shoulder_roll_joint` |
| 17 | `left_ankle_roll_joint` | `left_shoulder_yaw_joint` |
| 18 | `right_ankle_roll_joint` | `left_elbow_joint` |
| 19 | `left_shoulder_yaw_joint` | `left_wrist_roll_joint` |
| 20 | `right_shoulder_yaw_joint` | `left_wrist_pitch_joint` |
| 21 | `left_elbow_joint` | `left_wrist_yaw_joint` |
| 22 | `right_elbow_joint` | `right_shoulder_pitch_joint` |
| 23 | `left_wrist_roll_joint` | `right_shoulder_roll_joint` |
| 24 | `right_wrist_roll_joint` | `right_shoulder_yaw_joint` |
| 25 | `left_wrist_pitch_joint` | `right_elbow_joint` |
| 26 | `right_wrist_pitch_joint` | `right_wrist_roll_joint` |
| 27 | `left_wrist_yaw_joint` | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` | `right_wrist_yaw_joint` |

| 刚体位置误差的 body 索引 | 当前 PMT：body 名称 | PMT critic 输入切片 | 对比 checkpoint：tracked-body 名称 | 对比 critic 输入切片 |
| --- | --- | --- | --- | --- |
| 0 | `pelvis` | `[157:160]` | `pelvis` | `[100:103]` |
| 1 | `left_hip_pitch_link` | `[160:163]` | `left_hip_roll_link` | `[103:106]` |
| 2 | `right_hip_pitch_link` | `[163:166]` | `left_knee_link` | `[106:109]` |
| 3 | `waist_yaw_link` | `[166:169]` | `left_ankle_roll_link` | `[109:112]` |
| 4 | `left_hip_roll_link` | `[169:172]` | `right_hip_roll_link` | `[112:115]` |
| 5 | `right_hip_roll_link` | `[172:175]` | `right_knee_link` | `[115:118]` |
| 6 | `waist_roll_link` | `[175:178]` | `right_ankle_roll_link` | `[118:121]` |
| 7 | `left_hip_yaw_link` | `[178:181]` | `torso_link` | `[121:124]` |
| 8 | `right_hip_yaw_link` | `[181:184]` | `left_shoulder_roll_link` | `[124:127]` |
| 9 | `torso_link` | `[184:187]` | `left_elbow_link` | `[127:130]` |
| 10 | `left_knee_link` | `[187:190]` | `left_wrist_yaw_link` | `[130:133]` |
| 11 | `right_knee_link` | `[190:193]` | `right_shoulder_roll_link` | `[133:136]` |
| 12 | `left_shoulder_pitch_link` | `[193:196]` | `right_elbow_link` | `[136:139]` |
| 13 | `right_shoulder_pitch_link` | `[196:199]` | `right_wrist_yaw_link` | `[139:142]` |
| 14 | `left_ankle_pitch_link` | `[199:202]` | — | — |
| 15 | `right_ankle_pitch_link` | `[202:205]` | — | — |
| 16 | `left_shoulder_roll_link` | `[205:208]` | — | — |
| 17 | `right_shoulder_roll_link` | `[208:211]` | — | — |
| 18 | `left_ankle_roll_link` | `[211:214]` | — | — |
| 19 | `right_ankle_roll_link` | `[214:217]` | — | — |
| 20 | `left_shoulder_yaw_link` | `[217:220]` | — | — |
| 21 | `right_shoulder_yaw_link` | `[220:223]` | — | — |
| 22 | `left_elbow_link` | `[223:226]` | — | — |
| 23 | `right_elbow_link` | `[226:229]` | — | — |
| 24 | `left_wrist_roll_link` | `[229:232]` | — | — |
| 25 | `right_wrist_roll_link` | `[232:235]` | — | — |
| 26 | `left_wrist_pitch_link` | `[235:238]` | — | — |
| 27 | `right_wrist_pitch_link` | `[238:241]` | — | — |
| 28 | `left_wrist_yaw_link` | `[241:244]` | — | — |
| 29 | `right_wrist_yaw_link` | `[244:247]` | — | — |
