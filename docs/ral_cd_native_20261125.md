# C/D 两机增量实验：GPU 原生响应监督

本轮由用户明确要求进入 C/D 实现，停止扩展独立诊断队列。4090 不参与。历史数据、失败结果与冻结协议保留；旧六小时协议属于旧代码版本，不能用它验证本轮新代码。

本次只修改代码并完成静态检查，没有启动 GPU 试跑、训练或额外实验。下面的预算是预先指定的开发实验预算，不是已完成结果，也不保证组间优势。

## 两台机器训练什么

- **5080：** 使用 intervention01 的平滑 B（B_v005），种子固定为 501、505、509、513、517。每个父模型生成一份 GPU 响应数据，再分成 B_continue、C 两支，共 10 次训练。每支新增 2000 轮、10000 次世界模型更新，其中前 100 轮只更新世界模型，随后 1900 次策略更新。保留 WM 窗口 8、策略窗口 32、1024 个仿真环境。
- **3090：** 接收每个种子的完整交接包；计算校准结果、采集公共 fit 起点，更新 C_fixed8、C_fixed32、D、C_perm 四个策略分支，每支 200 次更新、批大小 64，世界模型和价值网络冻结。最多产生 20 个策略。D/C_perm 若在实际起点池上没有长短窗口变化，会明确标为 blocked，不伪装成完成；固定窗口对照仍执行。
- **统一评价：** 3090 对父 B、B_continue、C 以及成功完成的窗口分支，在原有六种参数条件、163 条动作上各评价一次。没有新增参数扫描或重复一致性诊断。D 的主对照是 C_perm 和同起点的 C_fixed32，不把初始 C 的 EMA 与窗口分支的 live 权重直接比较成因果收益。

新增环境转移、继承的 5000 轮父模型预算、GPU 标签生成转移数及耗时分别记录。上述是增量消融，不是从零训练的样本效率比较。实际用时尚未实测，不承诺在固定小时内全部完成。

## 数据与训练改动

GPU 响应对由原训练 `TrackingSim` 场景、相同参数、域随机化、1024 个 world 和四个 5 ms 物理子步生成。两支使用同一完整积分状态与相同 world 槽位，保留真实接触和求解器热启动。生成器使用冻结父 B 的 live 策略做 20 步预滚动，随后记录 32 步基准命令并施加正负扰动。物理计算操作在外层 CUDA 图中执行；该图不包含训练器的奖励、重置、回放写入和策略更新，不能称为整个训练图逐位相同。

另一次相同基准命令回放提供数值重复噪声估计，这是标签生成中的质量记录，不是另一个排障队列。所有原始块保留。训练只采样第 4 步前缀信号至少 0.01、至少为重复噪声 3 倍、且起点未终止的 fit 块。没有合格块时停止并保留数据，不静默退回 CPU 标签。

只沿用旧 manifest 的 source-session 划分，**不读取旧 CPU fit/cal/test NPZ 标签**。新数据只生成 fit/cal；test session 列表保留锁定，test 数据不生成。父策略已经使用过完整动作库，因此不能称为未见动作泛化。

C 保留在线回放前向损失，增加 4 步 Huber 响应项，初始设定如下：

- 世界模型 FP32，策略/价值网络保持父模型配置。
- 每 5 次世界模型更新计算一次 128 对响应；B_continue 也计算相同批次的响应诊断，但不接受其梯度。
- 响应权重上限 `1e-4`，250 轮线性增加；加权响应梯度范数最多为前向梯度范数的 10%，之后仍执行原梯度裁剪和优化器。
- 上述权重、梯度比例和窗口是预先指定的工程起点，不是已验证最优值。Muon 会进一步变换梯度，10% 上限不是参数更新量或控制收益的保证。
- 两支从相同 WM、live actor、EMA actor、critic、归一化统计开始，优化器、回放、重置统计与学习率计划同时重新开始。策略变化后各自采集的数据会分化，不能声称训练轨迹逐条相同。响应数据和响应采样随机流保持配对。

## D 的校准规则

窗口候选为 8/32；根据当前关节速度与左右脚踝高度构成的 16 个分箱分配。脚踝高度沿用项目 `CONTACT_Z`，是当前状态可计算的支撑代理，不是接触力测量，也不使用未来真实轨迹。

校准先在每个源 session 内取 90% 分位，再在 session 间取 90% 分位。每箱至少 10 个块、2 个源 session。所有前缀的相对响应 RMSE 都须不超过 **0.5**；分母是同前缀零响应预测器的误差。信号不足、重复噪声过大或终止起点的前缀按失败处理，不能通过删去困难块放宽门槛。0.5 是本轮固定的工程准则，不是经过实机标定的容差，也不构成策略回报保证。

没有通过长窗口条件的箱回退至 8，`fallback_bins` 和实际起点池未验证比例会记录；不把回退解释为已认证安全。实际起点池全短或全长时，D/C_perm 会标为 `blocked_degenerate_calibration`，不继续无信息的相同训练。

D/C_perm 的起点索引、初始权重和每次更新的窗口数量相同。当前实现仍执行稠密 32 步计算，只遮蔽短样本的后续贡献，**不能把 assigned_steps 减少写成实际 FLOPs 或耗时节省**。记录 computed_steps 与 wall_seconds；固定 8 步对照实际只计算 8 步。冻结 critic 的末端估值误差仍是本实验的限制。

## 安装本次代码

两台机器均把 `ral_cd_native_code_20261125.tar.gz` 和同名 `.sha256` 放到 Downloads，执行：

```bash
cd "$HOME/Downloads"
sha256sum -c ral_cd_native_code_20261125.tar.gz.sha256
tar -xzf ral_cd_native_code_20261125.tar.gz -C "$HOME/Projects/Supermimic-z01"
cd "$HOME/Projects/Supermimic-z01"
uv sync --locked
```

代码包不包含大模型、动作 NPZ 或机器人网格；两台机器继续使用现有项目资产与真实动作。不要只复制两个启动脚本，必须同步整个代码包。代码、资产、动作、参数和依赖版本在启动时冻结，两机交接时核对。

## 5080 启动

必须保留 `logs/ral_3h/intervention01_5080/` 原会话，包括五个完整种子的 `blocks/s*/B_v005/models/B_v005_models.tar.gz` 与 `status.json`。review 压缩包不能替代模型文件。程序核对此前审阅时冻结的归档及检查点哈希。

```bash
cd "$HOME/Projects/Supermimic-z01"
mkdir -p logs/ral_cd
nohup bash scripts/ral_5080_cd.sh --session cd01 \
  > logs/ral_cd/cd01_5080_console.log 2>&1 &
```

源会话放在其他目录时，在启动参数末尾增加 `--source /实际目录/intervention01_5080`。

每完成一个种子的 B/C，就输出：

```text
logs/ral_cd/cd01_5080/handoff/cd01_s501.tar.gz
logs/ral_cd/cd01_5080/handoff/cd01_s501.tar.gz.sha256
```

后续对应 s505、s509、s513、s517。每包包含父 B、B_continue、C 的完整 WM/actor/critic 检查点、元数据、GPU fit/cal 数据、积分快照、参数和配置。无需复制优化器文件，也无需复制过去 23.7 GB 的全部日志。

## 3090 交接与启动

把上述交接包和校验文件复制到 3090 的 `data/ral_cd_inbox/`，**不需要手动解压交接包**。大文件先复制为 `.part`，复制完成后改成 `.tar.gz`，最后复制 `.sha256`；不要让未完成的文件与校验文件同时可见。

```bash
cd "$HOME/Projects/Supermimic-z01"
mkdir -p data/ral_cd_inbox logs/ral_cd
nohup bash scripts/ral_3090_cd.sh --session cd01 \
  --inbox data/ral_cd_inbox --wait-hours 12 \
  > logs/ral_cd/cd01_3090_console.log 2>&1 &
```

3090 可以先启动，等待交接包时不做 GPU 训练；一个种子到齐即可开始，不必等五个全部训练完。若所有新包先下载到了 3090 的 Downloads，可复制已完成的文件：

```bash
cp "$HOME"/Downloads/cd01_s*.tar.gz data/ral_cd_inbox/
cp "$HOME"/Downloads/cd01_s*.tar.gz.sha256 data/ral_cd_inbox/
```

如果两台机器可互相 SSH，传输也可用现有的文件传输工具；脚本不假设远端用户名/IP，不自动连接或上传。

## 查看进度、继续与回传

```bash
# 在对应机器执行；只读状态，不运行模型
bash scripts/ral_5080_cd.sh --session cd01 --status
bash scripts/ral_3090_cd.sh --session cd01 --status
tail -n 40 logs/ral_cd/cd01_5080_console.log
tail -n 40 logs/ral_cd/cd01_3090_console.log
```

外层日志给出当前任务与它的 `jobs/.../attemptXX/console.log` 路径；训练迭代写在该任务日志中。

中断、空间不足或等待超时后，修复对应问题，在原启动命令中加 `--continue-session`。已完成任务会校验后复用；失败任务在新 attempt 目录从共同父模型重新开始，保留旧 attempt，不伪装成精确续训。不要同时启动两个相同 GPU 队列。

正常完成或停止时自动生成：

```text
logs/ral_cd/cd01_5080_review.tar.gz（以及 .sha256）
logs/ral_cd/cd01_3090_review.tar.gz（以及 .sha256）
```

回传这两个轻量 review 包、校验文件及各自 console.log 即可。两机保留原模型和交接包。`summary.json` 只汇总完整的种子内对照，负结果保留；没有 D 数据时不填零。
