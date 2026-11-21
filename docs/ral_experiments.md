# RA-L 七组实验执行说明

本文件和 `configs/ral/tasks.json` 是当前仓库的执行需求。论文附件中早期“仅 A0/A、48 小时”的运行单保留为历史资料。用户已选择完整七组；正式训练预算应在先导运行后冻结。

**当前默认入口已切换至 pilot02。** 使用本文末尾的 pilot02 指令；下列 pilot01 命令、导入结论和 v1 协议说明保留为历史记录，不应拿新源码继续使用旧协议。

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

最初提供的 `z01_lafan_v2_clean.zip` 含 163 个失效符号链接和筛选清单，包内实际 NPZ 文件数为 0；该旧包的检查记录为 `release/ral/motion_archive_status.json`。2026-11-21 已从新包 `z01_lafan_v2_clean_data.tar.gz` 导入并验证 **163 个真实 NPZ**，当前准备状态见 `release/ral/data_readiness.json`。以后从源机器打包时仍应解引用，例如在原工程运行：

```bash
tar -chzf z01_lafan_v2_clean_data.tar.gz -C data/motions z01_lafan_v2_clean
```

也可复制真实 `z01_lafan_v2/segments/*.npz` 并按 `selection.json` 选择对应 163 段。不要把链接文本改名为 NPZ。正式数据补齐后先核对文件、模型/关节顺序、50 Hz、数值及筛选来源。

当前已生成 `data/ral_pairs_v1/`（fit/cal/test 分别为 400/116/136 对，源动作会话分别为 23/7/7 个，窗口 32 步）和 `release/ral/protocol_v1.json`。准备日志为 `logs/ral_verification/prepare_real_data.log`；这批数据是 CPU MuJoCo 物理仿真配对数据。生成期间不要替换动作目录或重新解压旧 ZIP；生成器会检查失效链接、使用绝对路径，并在写出前验证动作和参数内容未变。现有 v1 产物可直接同步到另外两台机器，不需要重新运行 `prepare`。

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

## 三台电脑的 pilot01 先导任务（历史配置）

使用 `bash scripts/ral_three_machine.sh --help` 查看统一入口。脚本串行启动每台机器上的任务；任一阶段失败即停止，不覆盖既有训练目录。三机同步同一版本代码、模型资产、`uv.lock`、`release/ral/params.csv` 和真实动作库，再分别运行 `uv sync --locked`。GPU 型号在这里表示机器分工，不是自动硬件检测；三台机器的完整预算显存和运行速度仍需先导运行验证。

1. **3090：先准备公共数据，再运行窗口分组和统一评估。** 真实动作补齐后运行 `bash scripts/ral_three_machine.sh prepare`，生成一次 `data/ral_pairs_v1/` 和 `release/ral/protocol_v1.json`，将原样副本分发至另两台机器。配对生成使用 CPU MuJoCo；不要在三台机器分别生成并混用不同版本。
2. **4090：种子 1 的五组完整训练。** 运行 `bash scripts/ral_three_machine.sh train4090`，顺序为 A0 → A → A+ → B → C。
3. **5080：种子 2 的五组完整训练。** 运行 `bash scripts/ral_three_machine.sh train5080`，顺序为 A → A0 → A+ → B → C。每个种子的完整对照留在同一型号显卡，避免把跨显卡耗时差异当成方法收益。

每组使用 `configs/ral/common.json` 中的 **200 次迭代先导预算**，不是已收敛策略或正式论文预算。输出目录为 `logs/ral_<组名>_s<种子>_pilot01`，其中 A+ 的目录名写作 `Aplus`，C-perm 写作 `Cperm`。控制步长为 20 ms，固定策略窗口为 32 步（640 ms）。

将两台训练机的完整 C 目录分别复制到 3090 的对应位置。窗口校准误差容差尚未定值；必须依据拟合/校准数据和应用需求，在检查测试结果前明确，同一次比较采用一致规则。它是响应误差阈值，不能直接解读为关节角度弧度值。设置 `RAL_TOLERANCE` 为选定的正数后，在 3090 执行：

```bash
bash scripts/ral_three_machine.sh windows 1 "${RAL_TOLERANCE:?先设定校准容差}"
bash scripts/ral_three_machine.sh windows 2 "${RAL_TOLERANCE:?先设定校准容差}"
```

脚本为每个种子固定同一个 C 检查点与起点池，短/长窗口为 8/32 步（160/640 ms），每个窗口分支执行 100 次策略更新、批量 64、学习率 0.0001。世界模型和价值网络保持冻结。种子 1 顺序 D → C-perm，种子 2 顺序 C-perm → D。窗口分支不继续训练完整 C 模型，不能把其 100 次更新误报成独立的完整训练预算。

随后把 4090/5080 的全部十个训练目录复制到 3090，保留 `args.json`、检查点和日志。运行 `bash scripts/ral_three_machine.sh evaluate`，在同一台机器、评估种子 0 下，对七组 × 两个训练种子的最终检查点分别做 nominal/identified 评估，共 28 份结果 JSON。这是共享物理配置评估，不代表已经实现论文所需的独立 heldout 工况、实机测试或全训练过程的时间达标测量。

数据缺失时脚本会明确停止，不会切换到合成测试数据。失败后的新尝试使用新的 `RAL_ATTEMPT`（例如 `pilot02`），不要覆盖既有记录。若正式训练预算或配置改变，重新冻结新协议并三机同步；两个训练种子目前只用于先导对照，正式种子数和预算仍须预先确定。三种控制保持模型的等预算重辨识、异步/有记忆执行器和额外梯度替换对照仍不在这套已实现的七组启动命令内。

## 2026-11-21：pilot01 导入与下一步

4090 和 5080 返回的十个训练目录已导入本机 `logs/ral_*_s{1,2}_pilot01/`。每次训练完成 200 次迭代、577536 个物理环境步、1000 次世界模型更新和 200 次策略更新。60 个索引检查点的文件哈希、最终权重有限性、同种子配对初始化，以及本机代码/资产/参数/动作/响应数据与冻结协议的一致性均通过核验。可提交的摘要是 `release/ral/pilot01_review.json`；完整检查记录为 `logs/ral_verification/imported_pilot01_audit.json`。

**目前仍是先导训练完成，尚不能作为七组正式论文结果。** 在同一校准集的 116 对、7 个源会话上复核 32 步预测，B 的关节角 RMSE 为 0.06174 / 0.06032 rad，C 为 0.19277 / 0.15120 rad（依次为种子 1/2，FP32 推理）。C 的归一化响应 RMSE 为 113.342 / 113.087，零响应预测器为 112.905。这些数值是离线预测误差，不是闭环策略跟踪误差；归一化响应误差也不能解释为弧度。完整逐步曲线和 BF16 对照见 `logs/ral_verification/imported_pilot01_fit_cal.json`。两种推理精度结果接近，但这不能排除训练过程中的精度影响。

下一步先在 fit/cal 上诊断响应项与前向项的损失平衡、扰动信号和幅度稳定性，再用同初始化、同预算的小规模试验验证修改。当前 `response_weight=1.0` 不应直接沿用为正式设置；不能仅凭损失量级认定唯一原因。设置改变须使用新尝试和新协议，保留 pilot01，不覆盖 `protocol_v1.json`。

D/C-perm 暂未启动，窗口容差也尚未冻结；初始关节速度 RMS 大于 2 rad/s 的校准箱仅有 4 对，低于当前 10 对的覆盖要求。应先处理模型问题，再确定有依据的容差与短/长窗口覆盖，冻结 C 和共同起点池，执行两个窗口分支。不要为得到长窗口而任意放宽容差。当前检查只读取 fit/cal，未读取锁定测试结果，也未进行闭环评估。待正式预算、种子数和测试工况确定后，再生成统一评估结果。

进一步在最终检查点、全部 400 个 fit 块上分别求两项梯度：C 的响应梯度范数约为前向梯度的 536 / 570 倍（种子 1/2，BF16），两项梯度余弦均为负；FP32 下仍约为 574 / 538 倍。`response_weight=1.0` 配合总梯度裁剪并未平衡两个目标。当前证据首先指向响应监督压过前向拟合；这比仅比较损失标量提供了更直接的依据，但并非受控复训后的因果结论。记录见 `logs/ral_verification/pilot01_response_gradients.json`。

数据诊断发现，fit 中响应最强的 10% 块占零响应预测器平方误差总量的 77.9%。抽取响应最强的三个块及中位、最弱各一个块，共五块，CPU 原条件重放与保存的关节状态最大差异小于 9e-13；相同动作重复重放完全一致。在最强块上，epsilon 从 0.01 降至 0.005 后，32 步归一化响应相对差异为 1.107，方向余弦为 -0.214。这个抽样说明部分长时域响应对扰动幅度敏感，不能直接当作稳定局部导数；它不证明全部数据错误，也尚未区分接触切换、摩擦和求解器贡献。记录见 `logs/ral_verification/pilot01_pair_amplitudes.json`。后续先以同初始化、同预算验证响应权重候选，再验证幅度/时域有效性及 FP32 响应训练；共同数据或归一化规则改变时，A+/B/C 必须同步。训练源代码、原数据和冻结协议在本轮诊断中未修改。

## pilot02：响应目标修正后的执行入口

当前入口使用 `configs/ral/pilot02.json` 和 `release/ral/protocol_v2.json`；`common.json`、`protocol_v1.json` 及 pilot01 模型保留。训练源码改变后，旧协议会按设计拒绝新训练。两台机器需同步本次源码、配置和 v2 协议，继续使用原样的 `data/ral_pairs_v1/`、真实动作和参数表，无需重新生成或传输相同数据。新目录名为 `logs/ral_<组名>_s<种子>_pilot02`。

本轮已完成两个种子、八个候选共 16 次世界模型小对照，每次 120 次更新、批量 128，同种子共用初始化与实际采样索引。选定保守候选为 **`response_weight=0.0001`、`response_horizon=8`、`response_loss=mse`、`wm_amp=none`**。两个种子的 cal 关节角 RMSE 为 0.05769 / 0.05764 rad，不加响应项的对照为 0.05782 / 0.05766 rad。响应 RMSE8 平均改善仅约 0.007%，不能据此声称有效响应学习或论文收益。它目前只支持采用新配置继续先导验证；详见 `release/ral/pilot02_review.json`。窗口候选暂设 4/8 步，校准容差仍未定值。

代码检查包括 21 项单元测试、七组 GPU 合成流程测试，以及真实动作/完整网络/FP32/批量 256 的两次训练迭代启动检查。启动检查使用响应权重 0.001，只验证实际训练路径与日志；选定的 0.0001 另由上述双种子小对照验证。完整 200 次迭代的 pilot02 已由两台训练机完成，2026-11-22 复核结果见下文。

新增设置及其边界：

- `response_horizon` 只截取 C 的响应监督前缀；A+/B/C 的共同前向标签和正常化统计仍使用完整 32 步。原始配对数据不删除、不改写，诊断与校准仍能计算完整逐步误差。0 表示保留历史全跨度行为。
- `wm_amp=none` 让世界模型的拟合、策略推演和校准一致使用 FP32，并关闭 TF32；策略/价值网络的精度仍由 `amp` 控制。重新加载检查点时使用保存的设置。
- `response_loss=mse|huber` 和 `response_huber_delta` 支持显式稳健损失对照；Huber 不会改变原始误差统计。默认旧参数组合仍可复现旧目标，pilot02 的实际选择以配置为准。
- `train_metrics.csv` 增加 `forward_loss`、`weighted_response_loss`、`response_raw_mse`。后者为监督前缀内未经稳健损失变换的 MSE；两项加权损失之和应与总 `wm_loss` 一致。
- 新先导预算仍为 200 次迭代，世界模型批量统一设为 256，以容纳 FP32。所有五组都使用同一新配置并重新配对训练，不能把旧 A0/A/B 与新 C 拼成同预算比较。

在两台机器的项目根目录分别执行：

```bash
# 4090：种子 1，A0 -> A -> A+ -> B -> C
bash scripts/ral_three_machine.sh train4090

# 5080：种子 2，A -> A0 -> A+ -> B -> C
bash scripts/ral_three_machine.sh train5080
```

仅在以后再次更改配置或源码、且选定新的协议路径时，才运行 `bash scripts/ral_three_machine.sh freeze`。当前已分发的协议应直接使用，不能覆盖重建。可通过 `RAL_CONFIG`、`RAL_PROTOCOL`、`RAL_ATTEMPT` 指定新的配置/协议/目录；三者必须对应。

小规模目标对照入口如下，输出目录必须是新目录。它只训练世界模型，并用 cal 检查前向/响应误差，不更新策略、不读取 test，也不能充当完整 C 训练结果。实际采样索引、初始化哈希、预算、原始误差和每个候选的检查点均保留。

```bash
uv run python tools/ral_response_sweep.py \
  --plan configs/ral/response_sweep_v2.json --out logs/ral_response_sweep_repeat
```

修正后的误差是否达到论文要求，仍需后续独立评估验证。3090 收到新 C 后须先检查响应质量、校准覆盖和有依据的容差，再决定是否运行 D/C-perm；窗口候选从配置的 `window_candidates` 读取，不会自动放宽容差或把全部短窗当作有效窗口分配证据。

## 2026-11-22：pilot02 复核结果与下一步

`logs_4090_v2.zip` 和 `logs_5080_v2.zip` 已导入 `logs/ral_*_s{1,2}_pilot02/`，旧版本保留。十次训练均完成 200 次迭代、577536 个环境步、1000 次世界模型更新和 200 次策略更新。60 个索引检查点哈希、最终权重有限性、同种子配对初始化、冻结输入与新损失分解全部通过核验。完整训练复核记录为 `release/ral/pilot02_training_review.json`；`pilot02_review.json` 仍保留复训前的小规模候选选择依据，不用新结果改写历史选择。

在同一 cal 集的 116 对、7 个源会话上，按检查点保存的 FP32 设置复测，C 的 32 步关节角 RMSE 为 **0.05548 / 0.05551 rad**，同期 B 为 **0.05570 / 0.05576 rad**（种子 1/2）。归一化前向 MSE 则为 C 7.704 / 7.746、B 7.600 / 7.608。前向预测已恢复至 B 附近，pilot01 中 C 的明显退化得到控制；这并不代表响应目标已经有效，也不能把同时改变精度、批量、权重和监督时域的 pilot01/pilot02 差异归因于单个因素。

响应项仍未显示校准收益。C 的 fit 8 步归一化响应 RMSE 为 61.479 / 65.164，B 为 71.142 / 71.085，分别下降 13.58% / 8.33%；但 cal 上 C 为 **74.011 / 74.586**，B 为 **73.958 / 73.864**，零响应预测器为 73.893。C 在两个种子上分别比 B 差 0.072% / 0.977%。第 50、100、150、200 次迭代的同期比较也均未显示 C 的 cal 响应 RMSE8 优势。这与响应过拟合或泛化不足相符，尚不能认定唯一原因。

完整 32 步 cal 响应 RMSE 为 C 130.038 / 123.008，B 116.590 / 115.281。当前响应监督仅覆盖 8 步，而固定策略推演仍为 32 步，需区分短前缀拟合和长时域响应行为。上述响应指标经过输出尺度与扰动幅度归一化，不是弧度；关节角指标也是离线预测误差，不是闭环跟踪成绩。详细结果及逐步曲线见 `logs/ral_verification/imported_pilot02_fit_cal.json`，中间检查点见 `imported_pilot02_learning_curve.json`。本轮未读取 test 响应或运行策略测试；cal 已用于开发选择，不能把它当成独立最终测试。

后续分工：

1. **3090 先验证响应数据和泛化问题。** 针对 4/8 步前缀检查多种扰动幅度下的响应一致性，并在相同初始状态和控制时序下验证 CPU/GPU 物理响应差异。先前五块的重放只排除了这些样本的保存/重放错误，不能替代短时域与跨后端验证。随后在 fit/cal 做有限、同预算的 B/C 对照，保留共同前向标签和归一化，不直接增加当前训练轮数。
2. **4090、5080 保留本轮全部结果，暂不追加整套训练。** 待上述诊断支持新配置后，再使用新尝试和新冻结协议；不要覆盖现有模型或把不同版本混为同预算对照。
3. **D/C-perm 与统一评估尚待前置条件。** 当前未设置有应用依据的窗口容差，最快速度箱仍仅 4 对；不为产生长窗口而放宽阈值。响应有效性、覆盖和容差明确后，才冻结共同起点、C 世界模型/价值网络并执行两分支。正式预算、种子数、独立 heldout 工况与实机证据仍需补齐。

本轮仅导入模型、生成检查记录并更新任务状态；训练源码、数据、pilot02 配置与 `protocol_v2.json` 均保持原样。

## 三台机器的 8 小时任务队列

用户随后要求三机持续工作 8 小时。新增队列按 `configs/ral/eight_hour.json` 执行独立开发实验，替代上文“暂不追加训练”的近期安排。它从新初始化开始，不续写 pilot02；每个完整策略任务在自己的目录冻结配置和协议，不覆盖 `protocol_v1.json`、`protocol_v2.json`。**8 小时是整台机器这一队列的总时限，包含准备、编译、训练、检查和保存，并非每组各跑 8 小时。** 完成一项就继续下一项，最后预留 30 秒停止任务和写记录。遇到输入缺失、显卡不符、磁盘限额或连续三项失败会提前结束并记录原因。

三台机器的实际训练内容：

- **3090**：先从 fit 集按 8 步响应能量分位抽取 12 块，重放 0.25/0.5/1/2 倍扰动，检查 4/8/32 步响应；随后训练 B 世界模型对照和五个 C 目标候选，每项 1000 次 WM 更新、批量 128、FP32、1024×3 网络。种子从 10001 递增。同种子的全部候选共用初始化、归一化和实际抽样索引。这里不更新策略，也不能冒充完整 C 分组。
- **4090**：种子 1、3、5……，每个种子按 A0 → A → A+ → B → 五个 C 候选执行完整训练。每项 1000 次策略迭代、5000 次 WM 更新，WM 批量 256，其余共同设置继承 pilot02。每轮都有同种子的 B 对照。
- **5080**：种子 2、4、6……，按 A → A0 → A+ → B → 同样五个 C 候选执行，预算和共同配置与 4090 一致。后续轮次反转 C 候选顺序，减轻固定顺序影响。不同显卡的耗时不作为方法优劣证据。

五个 C 候选是原 pilot02 目标（权重 1e-4、8 步、MSE）、更小权重（1e-5、8 步、MSE）、更短监督（1e-4、4 步、MSE）和两个 Huber 目标（权重 1e-3 / 1e-2、8 步、delta=1）。所有候选保留完整 32 步前向标签与相同归一化。策略推演仍为 32 步。这是预先列出的目标对照与独立种子复验，尚不代表其中任何候选有效；判断结果不会擅自扩大训练预算或启动 D/C-perm。

开始前，三机同步同一份源码、`tools/ral_eight_hour*.py`、六份 `scripts/ral_*_8h.sh` / `scripts/ral_*_judge.sh`、`configs/ral/eight_hour.json`、`configs/ral/pilot02.json`、`uv.lock`、模型资产、`release/ral/params.csv`、真实动作目录及完整 `data/ral_pairs_v1/`。**无需复制旧检查点**，也无需重新生成动作或响应对。在各自项目根目录执行一次 `uv sync --locked`；入口使用 `--no-sync`，不会在训练中重新解析依赖。

推荐后台启动，终端断开后仍可运行。下面三条分别在对应机器执行，同一机器只执行自己的那一条：

```bash
mkdir -p logs

# 3090
nohup bash scripts/ral_3090_8h.sh --session night01 > logs/night01_3090_console.log 2>&1 &

# 4090
nohup bash scripts/ral_4090_8h.sh --session night01 > logs/night01_4090_console.log 2>&1 &

# 5080
nohup bash scripts/ral_5080_8h.sh --session night01 > logs/night01_5080_console.log 2>&1 &
```

默认 8 小时，可用 `--hours 1` 缩短；单次上限为 8 小时。用 `--dry-run` 仅查看任务计划，不占用 GPU。已有同名目录会被拒绝，下次使用 `--session night02`。同一可见 GPU 的队列有文件锁；默认 `CUDA_VISIBLE_DEVICES=0`。`--max-jobs` 和 `--allow-gpu-mismatch` 仅供明确的小规模验证，正常三机运行无需设置。

随时运行本机判断脚本，训练中也可以读取已经完成的项目：

```bash
bash scripts/ral_3090_judge.sh --sessions logs/ral_8h/night01_3090
bash scripts/ral_4090_judge.sh --sessions logs/ral_8h/night01_4090
bash scripts/ral_5080_judge.sh --sessions logs/ral_8h/night01_5080
```

每项训练结束都会自动生成 `analysis.json` 并更新队列的 `judgment.json`，不必手动触发才能继续。检查包括已完成预算、模型文件哈希、权重/训练指标有限性及配对条件；完整策略训练还检查 WM/策略确实更新、损失分解和训练器的非有限更新计数。A0/A 的判断限于训练完整性，不把训练回报当成独立闭环成绩。A+/B/C 在相同 fit/cal 集上计算前向、关节角和原始响应误差；Huber 候选仍以原始响应误差判断。

开发筛查要求：C 的 cal 前向 MSE、关节角 RMSE 不超过同期 B 的 1.10 倍；8 步响应 RMSE 至少比 B 和零响应预测器各低 1%；32 步响应 RMSE 不超过 B 的 1.10 倍；至少一半校准会话改善。同配置至少两个不同种子且全部通过才标记 `candidate_for_review`，其余为 `no_confirmed_gain`。**这些是可审计的开发筛查阈值，不是窗口校准容差、统计显著性或论文验收标准。** 失败候选不会自动加长训练；队列继续其余预定候选/新种子，下一步方法选择留在汇总之后。B 未完成时跳过该种子的 C，避免生成没有配对对照的比较。

输出目录是 `logs/ral_8h/night01_<显卡>/`：`status.json` 给出当前任务、进程、总截止时间和结束原因，`judgment.json` 给出配对判断，`prepare.log` 记录输入检查；每个 `jobs/<种子_候选>/` 保存任务、控制台日志、配置/协议（策略任务）、`run/` 检查点及 `analysis.json`。3090 的首项另有 `amplitudes.json`。原始训练和判断都保留，不能只回传通过的候选。正常终止保留 `last.pt`；若总时限在任务中途到达，停止整个子进程组，保留已原子写出的周期检查点，并将该项标为超时/未完成，排除出完整预算对照。没有承诺从被中断的策略任务精确续训；重开队列使用新 session。

每项开始前检查磁盘：空闲不足 15 GiB 或本队列已有产物达到 120 GiB 时不启动下一项；它不是文件系统硬配额，当前项仍可能增加占用。OOM、文件损坏和非有限值都留下失败记录，不偷偷修改批量或精度重试。若提前结束，先查看 `status.json` 和对应 `console.log`。8 小时内尽量持续完成有效任务，不以空转或忽略硬件/数据错误凑时长。3090 的 CPU 原条件重放完整性检查失败会直接停止；不同扰动幅度下的响应变化单独记录，不据此虚构合格容差。

结束后，将三机完整 session 目录汇总到 3090，再做合并判断（输出路径须为新路径）：

```bash
uv run --no-sync python tools/ral_eight_hour.py judge \
  --sessions logs/ral_8h/night01_3090 logs/ral_8h/night01_4090 logs/ral_8h/night01_5080 \
  --out logs/ral_8h/night01_combined.json
```

汇总仅合并模式、共同配置、源码/数据/工具哈希一致的结果；WM-only 与完整策略训练分开判断，重复种子不增加确认种子数，缺失/改动/超时记录不算合格结果。所有实验仍为 `formal_results=false`；不读取锁定 test 响应、不执行独立闭环测试、不自动生成窗口容差。CPU 重放检查也不能替代尚待完成的 CPU/GPU 响应一致性验证。

判断命令默认退出码 0 表示报告成功生成，不表示候选通过。自动化调用可追加 `--require-gain`，没有经不同种子确认的候选时返回 3；报告内容仍完整保存。队列的 0 表示正常达到时限/任务上限或队列完成，2 表示输入、资源、取消或运行失败，具体原因以 `status.json` 为准。

实现验证记录见 `release/ral/eight_hour_validation.json`：全项目 32 项测试通过，3090 上用真实数据完成了小预算 WM 对照及 A0/A/A+/B/C 训练流程检查，并实测总截止时间能够停止训练、保留有限检查点、排除不完整预算。尚未进行整夜 8 小时压力运行，也未在本轮远程启动 4090/5080。
