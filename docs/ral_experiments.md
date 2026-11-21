# RA-L 七组实验执行说明

本文件和 `configs/ral/tasks.json` 是当前仓库的执行需求。论文附件中早期“仅 A0/A、48 小时”的运行单保留为历史资料。用户已选择完整七组；正式训练预算应在先导运行后冻结。

## 实现与结论边界

本仓库是 **PartyMimic Z01 分支**，保留其仿真采样、世界模型和策略更新流程。不能将此处 A0 命名为“未经修改的原版 SuperMimic”。七组使用相同参考接口、奖励、控制步长和测试物理场景。原有 `--wm_structured` 只是速度/位置积分结构，**不等于**本次增加的执行器结构。

- **A0**：原始标称物理参数，原世界模型。
- **A**：替换可验证且物理含义一致的参数，保留 A0 的网络、输入和损失。
- **A+**：辨识物理参数；世界模型接收原始动作和执行器参数上下文，与 B 的网络输入宽度、容量一致。
- **B**：同 A+ 的参数、配对数据、采样种子和前向目标；动作经显式 PD、饱和、黏性阻尼和可微摩擦近似进入世界模型。
- **C**：同 B 的配对数据和前向目标，额外加入多步正负扰动响应损失。
- **C-perm**：从选定 C 检查点开始，冻结世界模型与价值网络，在固定起点池上更新策略；每次更新打乱 D 的窗口标签。
- **D**：同一 C 检查点、起点池、初始化、批次顺序和更新次数；根据独立校准集的响应误差选择短/长窗口。

A+/B/C 的世界模型只拟合共同冻结的配对数据，避免不同策略在线轨迹构成额外数据差异；在线采样仍用于策略起点。此设计是本工程的明确实验选择，论文方法/实验段应同步说明。D/C-perm 是独立的固定模型策略更新阶段，不继续世界模型和价值网络训练。每批两组的窗口直方图、分配步数和实际计算步数一致；短窗口在自己的末端 bootstrap，保留该值对状态的梯度。

**本次显式执行器实现限定为无记忆瞬时通道。**非零驱动延迟、响应滤波时间常数、异步反馈保持尚未等价接入，不应声称已经实现完整异步执行器世界模型。MuJoCo 的 `frictionloss` 是约束摩擦，B/C 使用 `tanh(qd / friction_speed)` 可微近似；`friction_speed=0.01 rad/s` 是声明的建模设置。必须用独立回放检验其响应误差，再决定是否需要有记忆模型。论文第一项贡献的连续 PD / 固定周期保持 / 记录事件保持三组等预算重辨识仍需要原始日志，不能用七组策略结果替代。

## 输入和参数合同

原始快照归档在 `release/ral/sources/joint_parameter_summary_20261120.json`，转换清单为 `release/ral/params.csv`，共 26 个关节 × 5 类参数。清单包含原值、辨识值、两组生效值、单位、来源哈希、启用状态和原因。

当前主比较启用七个关节的 armature、damping、frictionloss，共 21 项：左右髋俯仰、左髋侧摆、右髋偏航、左右膝、左踝俯仰。依据是所提供快照中的验证状态；原始实机记录尚未在本机复核。

- 左髋偏航：验证候选和运行值有差异，等待精确版本。
- 右髋侧摆、左踝侧摆：留出验证未改善，保留原值。
- 右踝俯仰、右踝侧摆：快照为左侧复用，保留原值。11 月 21 日右踝新候选与这份 11 月 20 日快照分开，不能混用。
- 上肢、腰、头：无新辨识量，保留原值。
- load_offset：属于试验边界的外部负载，未作为前馈增益或全身恒定负载移植。
- command_delay_sec：来自记录事件控制回放，未当作本项目连续 PD 的等效延迟。
- 所有组保留本项目相同的 Kp、Kd、动作尺度和关节限位；源表中的试验 PD 仅为来源信息。

辨识左右不对称，因此所有七组共同关闭镜像增强。A0/A 共同使用固定学习率调度，关闭依赖各自训练回报的提前衰减。原随机化分布保留；同种子的初始模型哈希记录于 `run.json`。停止后使用新尝试目录；原训练器恢复会重新填充经验池，不能作为严格逐步续训或与不中断组直接配对。

用户提供的 `z01_lafan_v2_clean.zip` 含 163 个失效符号链接和筛选清单，**实际 NPZ 文件数为 0**。归档检查见 `release/ral/motion_archive_status.json`。需要在持有原始数据的机器上解引用打包，例如在原工程运行：

```bash
tar -chzf z01_lafan_v2_clean_data.tar.gz -C data/motions z01_lafan_v2_clean
```

也可复制真实 `z01_lafan_v2/segments/*.npz` 并按 `selection.json` 选择对应 163 段。不要把链接文本改名为 NPZ。正式数据补齐后先核对文件、模型/关节顺序、50 Hz、数值及筛选来源。

## 可复现入口

在项目根目录执行。下列训练命令只有输入齐全时才会启动；默认 200 次迭代是先导预算。

```bash
uv run python tools/ral.py groups
uv run python tools/ral.py preflight --group A0
uv run python tools/ral.py probe --out logs/ral_probe_new

uv run python tools/ral.py train --group A0 --seed 1 --out logs/ral_A0_s1_pilot01
uv run python tools/ral.py train --group A --seed 1 --out logs/ral_A_s1_pilot01

# 从真实动作生成 CPU MuJoCo 正负扰动对；输出明确标注为仿真数据。
uv run python tools/ral.py pairs --motion_dir data/motions/z01_lafan_v2_clean \
  --horizon 32 --out data/ral_pairs_v1

uv run python tools/ral.py train --group A+ --seed 1 --responses data/ral_pairs_v1/manifest.json --out logs/ral_Aplus_s1_pilot01
uv run python tools/ral.py train --group B --seed 1 --responses data/ral_pairs_v1/manifest.json --out logs/ral_B_s1_pilot01
uv run python tools/ral.py train --group C --seed 1 --responses data/ral_pairs_v1/manifest.json --out logs/ral_C_s1_pilot01
```

在多机配对运行前冻结协议，并在每个训练命令后附加 `--protocol release/ral/protocol_v1.json`：

```bash
uv run python tools/ral.py freeze --motion_dir data/motions/z01_lafan_v2_clean \
  --responses data/ral_pairs_v1/manifest.json --out release/ral/protocol_v1.json
```

冻结文件记录训练配置、源码、资产、动作、参数、响应数据摘要，以及 nominal/identified 的共同评估清单；输入变化会拒绝继续使用旧协议。未提供响应集时只能冻结 A0/A。当前仍是先导预算，必须在正式试验前锁定足够的共同训练步数。严格“未见动作”结论还需要单独划分策略参考库；配对响应集的分割本身不代表策略没有见过同名参考动作。

配对生成器按原始动作名（去掉 `__seg...`）划分 fit/cal/test，至少六个独立源动作；同一源动作的片段不会跨集合。双方从相同完整 MuJoCo 积分状态和相同预运行历史出发。CPU 与训练 GPU 后端差异必须在正式实验中单独验证。

`ResponseData` 的 NPZ 格式为：`s0 [B,65]`、`a_prev [B,26]`、`xi [B,X]`、`a_plus/a_minus [B,H,26]`、`s_plus/s_minus [B,H,65]`、`epsilon [B]`、`session_id/block_id [B]`。状态为根位置、wxyz 四元数、世界线/角速度、26 关节角/速度。epsilon 是整段关节目标扰动的 L2 范数（rad），不是归一化动作量。manifest 指定关节顺序、步长、参数摘要、输出尺度和各集合文件哈希。训练拒绝 test 集及未经显式测试标记的合成测试数据。

窗口校准容差必须在看测试集之前按应用目标设定，下列 `TOLERANCE` 为必填数值：

```bash
uv run python tools/ral.py calibrate --checkpoint logs/ral_C_s1_pilot01/last.pt \
  --responses data/ral_pairs_v1/manifest.json --short 8 --long 32 \
  --tolerance TOLERANCE --out release/ral/window_calibration_s1.json
uv run python tools/ral.py capture-starts --checkpoint logs/ral_C_s1_pilot01/last.pt \
  --motion_dir data/motions/z01_lafan_v2_clean --horizon 32 --out data/ral_starts_s1
uv run python tools/ral.py window-train --group D --checkpoint logs/ral_C_s1_pilot01/last.pt \
  --motion_dir data/motions/z01_lafan_v2_clean --calibration release/ral/window_calibration_s1.json \
  --starts data/ral_starts_s1/manifest.json --out logs/ral_D_s1_pilot01
uv run python tools/ral.py window-train --group C-perm --checkpoint logs/ral_C_s1_pilot01/last.pt \
  --motion_dir data/motions/z01_lafan_v2_clean --calibration release/ral/window_calibration_s1.json \
  --starts data/ral_starts_s1/manifest.json --out logs/ral_Cperm_s1_pilot01
```

校准按初始关节速度 RMS 分箱，先汇总每个会话，再计算跨会话误差分位数；长窗口要求整个前缀达标，且至少两次独立会话、有足够高于信号下限的块。短窗未获验证的起点单列为 fallback。不能将低信号、少会话或全部选择相同窗口视为自适应窗口有效性的证据。

## 统一闭环评估

所有组均在相同的 nominal 和 identified 物理配置上评估，策略不获知测试配置。正式 heldout 工况需要辨识不确定区间或明确工程依据，目前未虚构范围。

```bash
uv run python -m partymimic.evaluate --run_dir logs/ral_A_s1_pilot01 \
  --ckpt logs/ral_A_s1_pilot01/last.pt --parameter_manifest release/ral/params.csv \
  --plant nominal --out logs/ral_A_s1_pilot01/eval_nominal.json
uv run python -m partymimic.evaluate --run_dir logs/ral_A_s1_pilot01 \
  --ckpt logs/ral_A_s1_pilot01/last.pt --parameter_manifest release/ral/params.csv \
  --plant identified --out logs/ral_A_s1_pilot01/eval_identified.json
```

输出逐回合完成/终止原因、规划与实际时长、全 26 关节 SSE/计数/RMSE，以及根部世界 XYZ 线速度 SSE/计数/每分量 RMSE。非有限预测保留为失败，不丢弃。按共同训练种子、预算、参考动作和工况配对，提前终止与误差共同报告。两个训练种子只能展示各自结果与范围，不能据此声称显著性。

## 验收和产物

每组输出 `args.json`、`run.json`、输入文件摘要、`initial.pt`（A0–C）、`train_metrics.csv`、检查点和索引；定期检查点包含世界模型及价值网络。窗口组另存批次起点索引和实际/分配推演预算。所有结果默认标记 `formal_results=false`；正式论文结果须经过任务清单中的协议冻结、来源复核和配对核验。

```bash
uv run python -m unittest discover -s tests -v
uv run python tests/ral_smoke.py /tmp/ral_unique_synthetic_test
```

第二条是显式合成数据的七组 GPU 接线测试，检查世界模型/策略更新、同种子初始化、冻结模型、起点与推演预算、重载和闭环入口。其数据与输出不得加入正式结果。
