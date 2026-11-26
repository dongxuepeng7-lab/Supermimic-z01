# InstinctMJ 项目分析报告

> **分析对象**：<https://github.com/project-instinct/InstinctMJ>
> **快照版本**：`main` 分支，最新提交 `6edbc19`（2026-09-01，"docs: document mjlab and instinct_rl versions"）
> **本地副本**：`_repos/InstinctMJ-main/`（372 个文件，18.8 MB 压缩包，解压后约 42 MB，其中 84 个 STL 网格占 41 MB）
> **分析方式**：全量文件清单 + 架构/配置/核心模块精读 + 规模与依赖统计。**未实际训练**（本机为 Windows，无 NVIDIA CUDA + MuJoCo Warp 运行时，项目本身也不支持 Windows）。

---

## 0. 一页速览（TL;DR）

| 维度 | 结论 |
|---|---|
| **一句话** | 清华 IIIS / 上海期智研究院 **Project Instinct** 生态的"环境侧"代码库，把原本基于 IsaacLab 的 InstinctLab 移植到 **mjlab + MuJoCo Warp**，用于人形机器人全身控制 RL 的训练与评测。 |
| **技术上是什么** | 一个建立在 `mjlab` 之上的 manager-based MDP 任务套件：运动参考（motion reference）子系统 + 参数化命令库 + 多奖励组 + 传感器/地形/监控体系。 |
| **规模** | 203 个 Python 文件，**36,119 行** Python；STL 资产 41 MB。 |
| **任务** | 注册 **16 个任务 ID**（7 个任务族 × train/play；perceptive 另有 OneMotion 变体）：locomotion、BeyondMimic tracking、whole-body shadowing、perceptive shadowing、perceptive VAE、perceptive HOI、parkour AMP。 |
| **算法边界** | 环境侧只负责 MDP；算法在外部库 **instinct_rl**（PPO / **AMP-WASABI** / MoE / DAgger 蒸馏 / VAE 蒸馏）。**只看 InstinctMJ 无法训练 AMP**。 |
| **工程成熟度** | 架构清晰、可替换性好、迁移保真度被反复对齐；但公开快照**没有测试、没有 CI、没有数据、没有实验结果**，且大量本机路径硬编码。 |
| **许可证** | **CC BY-NC 4.0（禁止商用）**；`instinct_rl` 为 Modified MIT。学术使用可以，商用/再分发需谨慎。 |
| **对你的价值** | 与你的 *Actuator Reserve Gating of AMP* 高度相关：它提供了一套**可作为实现参考**的人形 AMP 管线（G1、AMASS/LAFAN1、WASABI 判别器、执行器建模、监控指标），但**不是可直接对比的 baseline**（任务、平台、评测口径都不同，且无公开实验数据）。 |

---

## 1. 项目定位与生态

### 1.1 它在 Project Instinct 里的位置

Project Instinct 是清华大学交叉信息研究院（IIIS）与上海期智研究院的一个"人形全身控制"研究项目群，作者包括 Ziwen Zhuang、Shaoting Zhu、Mengjie Zhao、Hang Zhao。官网列出的代表工作：

- **Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostic Motions**（CoRL 2025，arXiv:2502.01465）
- **Deep Whole-Body Parkour**（在投，arXiv:2601.07701）
- **Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids**（在投，arXiv:2601.07718）

生态由四个代码库组成：

```
Project Instinct
├── InstinctLab      # 环境侧（IsaacLab 版，Isaac Sim 5.1 + IsaacLab 2.3.2）
├── InstinctMJ       # 环境侧（mjlab 版，本报告对象）★
├── instinct_rl      # 算法侧（PPO / AMP-WASABI / MoE / 蒸馏 / VAE）
└── instinct_onboard # 部署侧（ONNX 策略上机、观测/动作对齐）
```

InstinctMJ 的 README 明确写着：它是 **InstinctLab 的 mjlab-native port**，作为 Project-Instinct 的"environment side"。所以它不是一个独立的新项目，而是**同一套研究代码的第二个仿真后端**。

### 1.2 为什么要从 IsaacLab 迁到 mjlab

从代码与提交记录可以读出三条动机（部分为推断）：

1. **依赖更轻**：IsaacLab 依赖 Omniverse/Isaac Sim，安装重、版本锁死；mjlab 只依赖 MuJoCo + MuJoCo Warp + PyTorch，`uv sync` 即可。
2. **GPU 原生物理**：MuJoCo Warp 把 MuJoCo 的物理计算搬到 CUDA，4096 个环境并行是常规配置（locomotion 与 BeyondMimic 均默认 4096 envs）。
3. **直接访问 MuJoCo 原生数据结构**：对需要自写传感器（深度相机 raycaster、volume points、height scanner）和自定义地形（mesh、heightfield、CoACD 凸分解）的研究更友好。

代价是：**mjlab 的 API 仍在快速演进**，所以 InstinctMJ 把 `mjlab` 锁在 `1.6.0`，并以 editable 方式从 `../mjlab` 安装；`uv.lock` 锁定了 `mujoco 3.11.0 / mujoco-warp 3.11.0 / warp-lang 1.14.0 / torch 2.10.0 / instinct_rl 1.0.2`。

### 1.3 与相邻项目的边界

| 仓库 | 职责 | 与本项目的关系 |
|---|---|---|
| **mjlab** | manager-based API + MuJoCo Warp 仿真，Apache-2.0 | 直接依赖（`mjlab==1.6.0`，必须放在同级目录 `../mjlab`） |
| **InstinctLab** | IsaacLab 版环境侧 | 被移植的上游；行为对齐的参照物 |
| **instinct_rl** | 训练算法与网络模块 | 训练时依赖；**AMP 判别器实际在这里** |
| **instinct_onboard** | 真机部署 | 通过 `--export-onnx` 对接；Task README 专门强调关节顺序差异 |

---

## 2. 技术栈与运行前提

### 2.1 依赖与版本

| 组件 | 版本/约束 | 说明 |
|---|---|---|
| Python | `>=3.10,<3.14` | `instinct_rl` 自身 README 要求 3.12，实际建议 3.12 |
| mjlab | `==1.6.0` | **editable 安装自 `../mjlab`**，必须手动 clone 对应版本 |
| mujoco / mujoco-warp | `~=3.11.0`（锁定 3.11.0） | MuJoCo Warp 需要 NVIDIA GPU |
| warp-lang | `>=1.14.0`（锁定 1.14.0） | 自定义 CUDA kernel（采样、raycast、平滑） |
| instinct_rl | git 源，锁定 `1.0.2`（commit `3a28448`） | 算法与网络 |
| torch | 锁定 2.10.0 | — |
| 平台 | Linux x86_64 / macOS arm64 | **训练实际需要 Linux + NVIDIA GPU**；mjlab 官方说明 macOS 仅用于评估 |

### 2.2 安装路径

```bash
uv sync
uv run instinct-list-envs
```

前提是把 `mjlab`（1.6.0）clone 到同级目录 `../mjlab`，因为 `pyproject.toml` 里写的是：

```toml
[tool.uv.sources]
mjlab = { path = "../mjlab", editable = true }
instinct_rl = { git = "https://github.com/project-instinct/instinct_rl.git" }
```

### 2.3 与你的 Windows 环境的关系

**不能直接运行**，原因有三：

1. `pyproject.toml` 的 uv 环境只声明了 `darwin arm64` 与 `linux x86_64`；
2. MuJoCo Warp 需要 NVIDIA GPU + CUDA，Windows 不在官方支持路径内；
3. 部分配置在**导入时**就往 `/tmp/` 写文件（如 BeyondMimic 配置生成 motion selection YAML），Windows 没有 `/tmp` 会直接失败。

可行的用法：在 Windows 上做**源码阅读与设计借鉴**（本报告即为这种用法），真正复现需要一台 Linux + NVIDIA GPU 的机器。

---

## 3. 代码规模与结构

### 3.1 规模统计

| 指标 | 数值 |
|---|---|
| 文件总数 | 372 |
| Python 文件 | 203 |
| Python 代码行 | **36,119** |
| STL 资产 | 84 个，41 MB |
| 已注册任务 | 16 |
| 奖励函数/类定义 | 约 69 |
| 命令类定义 | 约 19 |
| 观测项定义 | 约 42 |
| 终止项定义 | 12 |
| 事件项定义 | 14 |
| 监控类定义 | 17 |
| 地形/配置类定义 | 约 47 |

### 3.2 子系统代码量

| 子系统 | 文件数 | 行数 | 职责 |
|---|---:|---:|---|
| `tasks/` | 59 | 9,992 | 任务族配置（环境 + RL 配置），代码量最大 |
| `envs/` | 34 | 6,723 | MDP 组件库（命令/奖励/观测/终止/事件/课程）+ Env 基类 |
| `terrains/` | 18 | 6,298 | heightfield / trimesh 地形、虚拟障碍、terrain importer |
| `motion_reference/` | 21 | 4,347 | 运动参考：数据类、buffer、manager、数据集加载 |
| `utils/` | 20 | 2,455 | 数学、缓冲、噪声、perlin、warp kernel、FK/IK |
| `scripts/` | 14 | 2,202 | 训练/播放/数据转换/可视化 CLI |
| `sensors/` | 21 | 1,733 | 噪声深度相机、grouped raycaster、volume points |
| `monitors/` | 4 | 1,107 | 训练监控与 actuator/轨迹统计 |
| `assets/` | 2 | 550 | G1 资产、执行器参数 |
| `rl/` | 4 | 435 | instinct_rl 配置 dataclass、vecenv 包装 |
| `managers/` | 3 | 227 | 多奖励管理器 |
| `visualization/` | 2 | 48 | 可视化 marker 配置 |

### 3.3 目录树（精简）

```
InstinctMJ-main/
├── pyproject.toml / uv.lock
├── README.md / CONTRIBUTING.md / LICENSE
└── src/instinct_mj/
    ├── envs/                 # InstinctRlEnv、InstinctScene、MDP 组件库
    │   └── mdp/{actions,commands,curriculums,events,observations,rewards,terminations}/
    ├── tasks/
    │   ├── registry.py       # 任务注册表（entry-point 机制）
    │   ├── locomotion/       # 速度跟踪
    │   ├── shadowing/        # beyondmimic / whole_body / perceptive / perceptive_hoi
    │   └── parkour/          # AMP + 深度 + 虚拟障碍
    ├── motion_reference/
    │   ├── motion_reference_manager.py   # 作为 Sensor 的运动参考服务
    │   ├── motion_buffer.py              # Dataset 抽象
    │   ├── motion_files/{amass,aistpp,omomo,emberUcb,terrain}_motion.py
    │   └── motion_generators/stay_still.py
    ├── sensors/{noisy_camera,grouped_ray_caster,volume_points}/
    ├── terrains/{height_field,trimesh,virtual_obstacle}/
    ├── monitors/             # 17 个监控项
    ├── rl/                   # alive 配置 + vecenv 适配
    ├── assets/unitree_g1.py  # G1 29-DoF 模型与执行器参数
    └── scripts/              # train/play/list-envs + 数据管线
```

### 3.4 值得注意的规模特征

- **配置代码占比极高**：`tasks/` 近 1 万行，大量是 dataclass 配置的显式展开（如 parkour 配置单文件 1,007 行、perceptive 943 行）。这是 IsaacLab 风格的典型特征：可读性好，但重复度高。
- **少数文件超大**：`shadowing_command.py` 1,715 行、`mesh_terrains.py` 1,625 行、`hf_terrains.py` 1,132 行、`motion_reference_manager.py` 1,117 行、`monitors.py` 1,092 行。这些是最难维护的部分。
- **注释密度高**：`NOTE` 类注释 120 处，大量解释"为什么这样写才能和 IsaacLab 对齐"，这实际上是**迁移正确性的证据链**。

---

## 4. 架构总览

### 4.1 分层图

```
┌──────────────────────────────────────────────────────────────┐
│ tasks/<family>/config/g1/__init__.py                          │
│   register_instinct_task(task_id, env_cfg, play_cfg, rl_cfg)  │
└───────────────────────────┬──────────────────────────────────┘
                            │ _REGISTRY（含 env/vecenv/runner entry point）
┌───────────────────────────▼──────────────────────────────────┐
│ scripts/instinct_rl/train.py  |  play.py        （tyro CLI）   │
│   load_env_cfg / load_env_cls / load_vecenv_cls / load_rl_cfg │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ InstinctRlEnv（继承 mjlab.ManagerBasedRlEnv）                  │
│   InstinctScene → Simulation(MuJoCo Warp)                     │
│   Managers: Action / Observation / Command /                  │
│             MultiReward / Termination / Event / Curriculum /  │
│             Monitor                                           │
└───────┬───────────────────────────────────┬──────────────────┘
        │                                   │
┌───────▼──────────────────┐   ┌────────────▼─────────────────────┐
│ MotionReferenceManager   │   │ Sensors                          │
│  (伪装成 Sensor 的服务)   │   │  NoisyGroupedRayCasterCamera     │
│  → MotionBuffer(Dataset) │   │  RayCastSensor(height_scanner)   │
│  → AMASS/LAFAN1/OMOMO    │   │  VolumePoints / ContactSensor    │
└──────────────────────────┘   └──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ InstinctRlVecEnvWrapper → instinct_rl.OnPolicyRunner          │
│   policy/critic 观测分组打包；多奖励堆叠为 (N, num_rewards)     │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 任务注册：可替换的 entry-point 设计

`tasks/registry.py` 的设计值得单独说：每个任务登记的是**工厂函数 + 类路径字符串**，而不是直接 import 类。

```python
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-v0",
    env_cfg_factory=G1ParkourAmpEnvCfg,
    play_env_cfg_factory=G1ParkourAmpEnvCfg_PLAY,
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)
```

默认入口是 `"instinct_mj.envs:InstinctRlEnv"`、`"instinct_mj.rl:InstinctRlVecEnvWrapper"`，但下游仓库可以换成自己的 env/vecenv/runner，而不需要改这个包。提交历史显示这是最近一次重构（`refactor: route swappable components through cfg.class_type`）的成果，目的就是"让下游仓库能替换任意组件"。

**评价**：这是整个仓库最"产品化"的设计。对研究代码来说，能做到 env/runner 可插拔、配置与实现解耦，比多数实验室代码库规范。

### 4.3 Env 层扩展

`InstinctRlEnv` 相对 mjlab 只做了四件事：

1. 用 `InstinctScene` 替换默认 Scene，让 `terrain.cfg.class_type` 生效（支持自定义 terrain importer）；
2. 路由**多奖励组**：如果 `cfg.rewards` 是 `MultiRewardCfg`，先临时清空交给父类，再替换为 `MultiRewardManager`；
3. 挂载 `MonitorManager`，并在 `step()`/`_reset_idx()` 中把监控数据写入 `extras`；
4. 缓存 body 线加速度（供监控项使用）。

`MultiRewardManager` 的关键能力是：**一次计算多组奖励**，返回 `dict[group] -> (num_envs,)`，再由 vecenv wrapper 堆叠成 `(num_envs, num_rewards)`。这对应 instinct_rl 的"多奖励/多 critic"训练（例如 MoE 或多判别器场景）。

### 4.4 MDP 组件库

| 组件 | 规模 | 特点 |
|---|---|---|
| 命令（commands） | 约 19 类 | 见 5.1，是整个仓库最有特色的部分 |
| 奖励（rewards） | 约 69 个 | tracking / imitation / regularization / volume penetration / contact 等 |
| 观测（observations） | 约 42 个 | 本体感知、参考状态、深度图、高度扫描、参考-as-state |
| 终止（terminations） | 12 个 | 参考偏离、摔倒、出界、数据集耗尽 |
| 事件（events） | 14 个 | 域随机化、重置、参考匹配场景、bin 失败平滑 |
| 课程（curriculums） | 多个 | BeyondMimic 自适应采样、地形等级 |

奖励函数分两类命名，很值得注意：

- **`*_tracking_*`**：跟踪"当前时刻"参考（经典 mimic）；
- **`*_imitation_*`**：模仿"未来若干帧窗口"的参考（BeyondMimic 风格，带 `in_relative_world_frame`、`combine_method="mean_prod"` 等参数）。

这种"tracking vs imitation"的语义区分在代码里是一等公民，不是临时起意。

---

## 5. 核心机制详解

### 5.1 Motion Reference：把"参考动作"做成一个 Sensor

这是 InstinctMJ 最核心、也最值得借鉴的设计。

**问题**：motion tracking / shadowing / AMP 都需要"在时刻 t 查询参考运动的状态（关节位置、速度、根位姿、连杆位姿）"，而且不同任务需要不同子集（只要位置？只要关节？要未来 10 帧？）。

**方案**：`MotionReferenceManager` 继承 `mjlab.sensor.Sensor`，作为一个**场景级服务**挂在 `scene["motion_reference"]` 上；命令、观测、奖励、终止都从它取数据。

数据模型（`motion_reference_data.py`）：

```
MotionSequence   : 整条轨迹的原始数据（joint_pos/vel, base_pos/quat/vel, link_pos/quat, framerate, buffer_length）
MotionReferenceData : 当前时刻、多帧窗口的参考（num_envs, num_frames, ...）
                     每个通道都有配套的 mask（joint_pos_mask / link_rot_mask / base_pos_height_mask ...）
                     外加 time_to_target_frame 与 validity
MotionReferenceState: 单帧状态（用于 reset 时把机器人摆到参考姿态）
```

**关键设计点**：

1. **`num_frames` + `frame_interval_s`**：一次返回若干未来帧，`data_start_from` 决定第 0 帧是"当前时刻"还是"下一帧"。BeyondMimic 用 1 帧，parkour AMP 用 **10 帧 / 0.02 s**（即 0.2 s 窗口）。
2. **per-channel mask**：每个通道独立掩码，意味着可以"只跟踪下半身关节""只跟踪高度""忽略某些连杆"。这让同一套命令库能表达全身跟踪、部分跟踪、误差驱动跟踪等不同任务。
3. **aiming frame**：`aiming_frame_idx` 根据当前时间与 `frame_interval_s` 计算"瞄准帧"，并给出 `time_to_aiming_frame`；这套时间语义是为了复刻 IsaacLab 的 keyframe 行为。
4. **lazy update**：`update(dt)` 只推进时间戳，真正的数据刷新在**访问 `data` 时**触发（`_update_outdated_buffers`），并按每个 env 的 `update_period` 判断是否过期。好处是：不访问的 env 不付计算成本；坏处是：时序语义依赖访问顺序，调试时容易困惑。
5. **`_is_outdated` + `TimestampedBuffer`**：复位、切换、更新都通过"打脏标记"驱动，而不是每步全量重算。

**评价**：这是一个"以数据为中心的 MDP"设计——把参考运动抽象为可查询的服务，命令只是它的投影。相比在每个 command 里各自维护一条轨迹，这种设计让"命令子集化、观测复用、多任务共享"变得自然。

### 5.2 数据加载与重定向管线

`motion_reference/motion_files/` 支持多种数据格式：

| 类 | 数据源 | 用途 |
|---|---|---|
| `AmassMotion` | AMASS npz（含 LAFAN1 经 GMR 重定向后的 npz） | 主力：shadowing / parkour |
| `AistppMotion` | AIST++ | 舞蹈动作 |
| `OmomoMotion` | OMOMO（人-物交互） | perceptive HOI |
| `EmberUcbMotion` | Ember/UCB 数据 | 备选 |
| `TerrainMotion` | 与地形配对的动作 | 地形匹配任务 |
| `stay_still`（generator） | 在线生成"站定"参考 | 生成式 motion buffer |

以 `AmassMotion` 为例，它做了这些事：

1. 扫描目录，按 `filtered_motion_selection_filepath`（YAML）过滤；
2. 用 `pytorch_kinematics` 构建机器人链，做**重定向/前向运动学**，得到 link 位姿；
3. 支持**双线性插值**（`motion_interpolate_bilinear`）与前后向速度估计（`frontbackward` / `frontward`）；
4. 把轨迹切成 `motion_bin_length_s` 的 bin，用 **Warp kernel** 做 CDF 采样选择"从哪一帧开始"；
5. 支持**多进程分割**（`mp_split_method="Even"/"Segment"`），每个 rank 只加载自己那部分轨迹，避免 4096 envs × 大量动作的内存爆炸；
6. 支持**左右对称增强**（joint mapping + reverse buffer + link mapping），按 env 随机翻转。

配套的数据脚本说明这条管线是"从视频到训练"的全链路：

```
PHALP（4D-Human 视频跟踪） → phalp_to_amass.py → AMASS npz
AMASS/LAFAN1 → GMR（General Motion Retargeting）→ GMR_to_instinct.py → instinct npz
                → amass_filter.py（滤掉不可行动作）→ 训练
                → motion_matched_metadata_generator.py → metadata.yaml（动作↔地形/物体配对）
```

**评价**：这是仓库里"论文级"价值最高的一部分。多数开源 RL 环境只给"已经处理好的 npz"，而这里把**过滤、重定向、配对、可视化**都开源了——这也解释了为什么它能支撑"Deep Whole-Body Parkour / Hiking in the Wild"这类需要大规模动作-地形配对的工作。

### 5.3 自适应采样与课程（BeyondMimic 复刻）

`curriculums/motion_reference.py` 实现了 BeyondMimic 的 **自适应 bin 采样**：

- 把每条动作切成 1 s 的 bin；
- 每个 env 失败时，记录失败发生在哪个 bin；
- 用核 `[1, λ, λ²]`（λ=0.8）对失败计数做平滑，再归一化成采样概率；
- 保证 10% 的均匀探索（`adaptive_uniform_ratio=0.1`）；
- 用 Warp kernel 做原子累加与平滑（`update_current_fail_counters` / `compute_smoothed_probs`）。

此外还提供了变体 `BeyondConcatMotionAdaptiveWeighting`：把所有动作的 bin 当成一条大动作统一归一化，适合"多动作拼接"场景。

**评价**：这是"用课程学习解决 motion tracking 中难帧欠采样"的标准做法，实现完整，且把统计量（采样熵、top-1 概率）作为指标回传，便于判断训练是否崩掉。你的论文若涉及 AMP 参考分布采样，这里的 bin 化 + 自适应权重是可直接借鉴的工程实现。

### 5.4 执行器建模（与你的论文最相关之一）

`assets/unitree_g1.py` 用 **BeyondMimic 的 principles** 建模 G1 执行器：

| 参数 | 做法 |
|---|---|
| 转子惯量 → armature | 按电机型号给出 `ROTOR_INERTIAS_*`，结合减速比得到等效 armature |
| PD 增益 | `kp = armature × (2π × 10 Hz)²`；`kd = 2 × damping_ratio × armature × ω`，damping_ratio = 2.0 |
| 力矩上限 | 5020: 25 N·m；7520-14: 88 N·m；7520-22: 139 N·m；4010: 5 N·m（双电机翻倍） |
| 关节速度上限 | 显式维护 `G1_29DOF_JOINT_VELOCITY_LIMITS`（MJCF 不带 URDF 的 velocity 字段） |
| 通信延迟 | 每回合采样一次 `delay_min_lag=0..1`，整个 episode 保持；用"极大 `delay_update_period` + 关闭 `delay_per_env_phase`"来复刻 IsaacLab `DelayedPDActuator` 的语义 |

第 5 条尤其值得注意：mjlab 的 `DelayBuffer` 默认每个物理步重采样延迟，而 IsaacLab 是"每回合采样一次、保持到底"。作者用了一个**语义 hack**（把更新周期设成 1e6 步，让第一次采样后永不更新）来对齐，并在注释里写清了原因。提交历史里也有专门的修复提交（"Restore IsaacLab-equivalent actuator delay..."）。

**对你的意义**：你的论文核心变量是"执行器不确定性 + 扭矩余量"。InstinctMJ 这套"armature→kp/kd→力矩上限→延迟"的链条，是一个现成的、经过工程对齐的执行器参数来源；尤其 `kp/kd 由自然频率与阻尼比反推` 的做法，可以直接作为你"标定执行器"基线的对照。

另外，`monitors/monitors.py` 里的 `TorqueMonitorSensor` / `ActuatorMonitorTerm` 会记录**每关节力矩的均值/最大/最小**并画图。虽然它监控的是"施加力矩"而非你定义的"饱和前余量"，但作为训练期诊断工具（判断策略是否长期贴着力矩上限）可以直接复用。

### 5.5 感知仿真：噪声深度相机

`perceptive` 任务不依赖渲染，而是用 **raycaster 相机**直接生成深度图，再叠加一套噪声管线模拟 RealSense 类深度相机的真实缺陷：

| 噪声 | 作用 |
|---|---|
| `DepthContourNoise` | 深度不连续处的"飞点"（轮廓噪声） |
| `DepthArtifactNoise` | 随机矩形缺失块 |
| `RangeBasedGaussianNoise` | 距离相关高斯噪声 |
| `DepthSkyArtifactNoise` | 天空/远距伪影 |
| `GaussianBlurNoise` | 模糊 |
| `StereoTooCloseNoise` | 双目过近失效 |
| `CropAndResize` / `DepthNormalization` | 裁剪、缩放、归一化 |

相机配置（perceptive shadowing）：48×27（480×270 下采样 10 倍）、fovy 58°、安装在 torso、`distance_to_image_plane`、归一化到 0–2 m、裁剪到 18×32 后送入网络。

parkour 任务更进一步：64×36 深度、**37 帧历史**、按 `delayed_frame_ranges=(0,1)` 模拟通信/处理延迟、`history_skip_frames=5` 抽帧，最终给策略 8 帧。

**评价**：这是"sim-to-real 感知"里少见的细致实现。噪声模型不是随便加高斯，而是按真实深度相机的失效模式逐项建模。对做 perceptive humanoid 的工作，这部分是可直接复用的资产。

### 5.6 地形、虚拟障碍与 volume points

地形系统规模很大（18 文件 / 6,298 行 / 约 47 个类），支持：

- **heightfield 地形**：平地、坡、楼梯、波浪、perlin 起伏、tilted ramp 等；
- **trimesh 地形**：任意 mesh、运动匹配地形（`MotionMatchedTerrain`）；
- **terrain importer**：`hacked_generator` 模式可以接管 mjlab 的地形生成，插入自定义地形；
- **虚拟障碍**：从地形 mesh 中提取"边缘"，用 DBSCAN 聚类 + Warp raycast 生成圆柱障碍（`edge_cylinder.py`），再注册给 `VolumePoints` 传感器；
- **volume points**：在脚部/腿部体内布置点阵，检测这些点是否穿透地形/障碍，作为**穿透惩罚**（`volume_points_penetration`）。

parkour 的奖励里，`feet_at_plane`（用左右脚 height scanner 测量脚底与地面距离）、`volume_points_penetration`、`feet_slide`、`feet_flat_ori` 共同构成了"脚部接触质量"的完整约束。

**评价**：这套"虚拟障碍 + volume points"是 Deep Whole-Body Parkour 的关键工程：它让策略在**没有真实碰撞几何**的情况下也能感知"脚下边缘/台阶沿"，比单纯 height scan 更细粒度。

### 5.7 监控体系（17 个监控项）

`monitors/` 是很多开源环境缺失的一环。这里包括：

- `MotionReferenceMonitorTerm`：数据集统计（采样熵、top-n 采样动作）；
- `ShadowingPosition/Rotation/JointPos/JointVel/LinkPos`：与参考的逐项误差（可按 keyframe 阈值判定）；
- `ShadowingProgressMonitorTerm` / `ShadowingVelocityMonitorTerm`：跟踪进度与速度；
- `ActuatorMonitorTerm` / `TorqueMonitorSensor`：关节力矩、功率；
- `RewardSumMonitorTerm`、`BodyStatMonitorTerm`、`JointStatMonitorTerm`。

监控数据进入 `extras`，由 instinct_rl 记录。**对你的论文**：这类"逐项误差 + 力矩"监控正是做消融实验时最需要的训练期诊断，值得借用其字段设计（mean/max/min + keyframe 阈值）。

---

## 6. 任务族逐个解析

### 6.1 总表

| 任务族 | Task ID（train / play） | 目标 | 策略输入 | 算法 | 场景 |
|---|---|---|---|---|---|
| Locomotion | `Instinct-Locomotion-Flat-G1-v0` | 平面速度跟踪 | 本体感知 + 速度指令 | PPO | 平地，4096 envs |
| BeyondMimic | `Instinct-BeyondMimic-Plane-G1-v0` | 单条动作全身跟踪 | 参考关节/位姿 + 本体感知 | PPO | 平地，LAFAN1 |
| Whole-Body Shadowing | `Instinct-Shadowing-WholeBody-Plane-G1-v0` | 多动作全身 shadowing | 同上 + 更多命令 | PPO | 平地 |
| Perceptive Shadowing | `Instinct-Perceptive-Shadowing-G1-v0` | 深度感知下的 shadowing | 深度图 + 本体感知 + 参考 | PPO + Conv2d encoder | 运动匹配地形 |
| Perceptive VAE | `Instinct-Perceptive-Vae-G1-v0` | 深度策略蒸馏到本体策略 | 本体感知（学生） | VAE Distill（TPPO） | 同上 |
| Perceptive HOI | `Instinct-Perceptive-HOI-Shadowing-G1-v0` | 人-物交互 shadowing | 深度 + 本体 + 参考 | PPO | OMOMO 物体 |
| Parkour AMP | `Instinct-Parkour-Target-Amp-G1-v0` | 复杂地形速度/姿态控制 | 深度 + 本体历史 + 速度指令 | **WASABI AMP** + MoE | 程序化地形 + 虚拟障碍 |

### 6.2 Locomotion（基线）

- 动作：29 关节位置目标，`scale=0.5`；
- 观测：`base_ang_vel`、`projected_gravity`、速度指令、`joint_pos_rel`、`joint_vel`（带强噪声）、上一动作；critic 额外有 `base_lin_vel`；
- 奖励：速度跟踪（exp）+ 双脚腾空时间 + 惩罚滑步/姿态/关节偏差/力矩/加速度 + 终止惩罚（-200）；
- 终止：超时、姿态翻转、接触；
- 这是标准的 IsaacLab 速度跟踪配方，可作为"其他任务是否训练正常"的 sanity check。

### 6.3 BeyondMimic（单动作跟踪）

- 命令：位置/旋转/关节位置/关节速度四路参考命令，`realtime_mode=True`、`current_state_command=False`；
- 观测：参考四件套 + 本体感知；critic 额外给 link pos/rot（tannorm）；
- 奖励：base/link 的 pos/rot/lin_vel/ang_vel 的 **Gaussian imitation**（`mean_prod` 聚合）+ action_rate + joint_limit + undesired_contacts；
- 终止：根位置偏离 0.25 m（仅高度）、投影重力偏差 0.8、末端连杆偏离、数据集耗尽、出界；
- 课程：自适应 bin 采样；
- 域随机化：摩擦、关节零位、质心、推力。

这是 BeyondMimic 的忠实复刻（连奖励权重 0.5/0.5/1/1/1/1 与 std 0.3/0.4/0.3/0.4/1.0/3.14 都对应）。**注意**：它是"跟踪单条动作"，不是 AMP，不要和你的工作混淆。

### 6.4 Whole-Body Shadowing

与 BeyondMimic 同源，但命令更多（额外 `position_b_ref_command`），观测改为 `projected_gravity + base_ang_vel`（**不给 base_lin_vel**，更接近真机可观测性），奖励与终止基本一致。可以理解为"BeyondMimic 的多动作/可部署版本"。

### 6.5 Perceptive Shadowing / VAE / HOI

**Perceptive Shadowing**：

- 场景：`hacked_generator` + `MotionMatchedTerrain`，动作与地形通过 `metadata.yaml` 配对；
- 传感器：深度相机（噪声管线）、height scanner、contact、motion reference；
- 动作：29 关节位置，`scale=0.5`；
- 观测：深度图（18×32）+ 本体历史 8 帧 + 参考；
- 策略：`EncoderActorCritic` + Conv2d 深度编码器（channels [32,32]、输出 32 维）；
- 特点：**深度与动作-地形配对**是核心，策略要学会"看到台阶/障碍再决定落脚"。

**Perceptive VAE**：

- 教师：带深度编码器的策略（`EncoderActorCritic`），从 checkpoint 加载（配置里 `teacher_logdir` 是作者本机路径，需改）；
- 学生：`EncoderVaeActorCritic`，VAE latent 16 维，输入是深度编码器的 latent，辅助重建本体状态；
- 算法：`VaeDistill`（`using_ppo=False`，`teacher_act_prob=0.2`，KL 系数 1.0）；
- 目的：把"看深度的策略"蒸馏成"只看本体感知的策略"，部署时不需要深度相机（或降低对深度的依赖）。

**Perceptive HOI**：

- 数据：OMOMO（人-物交互），6 类物体（落地灯、大箱子、白椅子、垃圾桶、小桌子、行李箱）；
- 物体以 `mocap=True` 的 body 挂进场景，按动作的物体轨迹驱动；
- 机器人需要"看到物体并与之交互"，是 shadowing 的交互扩展。

**评价**：这三条线构成"感知 → 蒸馏 → 交互"的完整叙事，对应 Project Instinct 的论文群。工程上最复杂的是 HOI（物体 mesh、mocap 驱动、地形禁用）。

### 6.6 Parkour AMP（与你最相关）

**环境**（`g1_parkour_target_amp_cfg.py`）：

- 2048 envs，episode 20 s，decimation 4；
- 地形：程序化 rough terrain（perlin、方坑、金字塔台阶、箱子、mesh 箱子等）+ 虚拟边缘障碍；
- 传感器：双脚 contact、躯干 contact、非脚部 contact、leg volume points、左右脚 height scanner、**噪声深度相机（64×36，37 帧历史，延迟 0–1 帧）**、motion reference；
- 命令：`PoseVelocityCommand`（速度 + 朝向），且**按地形类型给不同速度范围**（如 `perlin_rough_stand` 要求站定、`pyramid_stairs` 要求前进）；
- 动作：29 关节位置，BeyondMimic action scale，delayed actuator；
- 策略观测：本体历史 8 帧 + 深度 8 帧 + 速度指令；
- 奖励：速度跟踪 + 存活 + 站立 + 脚部接触质量 + volume 穿透 + 力矩/加速度/动作率正则 + 安全约束（关节限位、速度限位、力矩限位、非期望接触）；
- 终止：超时、出界、躯干接触、姿态翻转、根高度过低、数据集耗尽；
- 课程：按速度跟踪成功率提升地形等级。

**AMP 部分**：

- 观测分组 `amp_policy`（真实状态）与 `amp_reference`（参考状态），各 5 项：`projected_gravity`、`joint_pos_rel`、`joint_vel`、`base_lin_vel`、`base_ang_vel`，**history_length=10 并展平**；
- 参考状态由 `*_reference_as_state` 系列函数从 `motion_reference` 取出；
- 算法：`WasabiPPO`，判别器 MLP `[1024, 512]`，奖励类型 `quad`，系数 0.25，MSE 损失，梯度惩罚 5.0，logit 权重衰减 0.04，判别器 AdamW lr 1e-4。

**这正是你论文需要的"AMP 环境侧接口"的完整样例**：如何把参考运动变成判别器输入、如何与 PPO 奖励相加、如何做多帧历史。区别在于：**它是 terrain-conditioned 的速度/姿态控制，不是 phase-conditioned 的步态风格控制**；判别器在 instinct_rl 里，环境只提供 state 组。

### 6.7 任务族设计上的一个观察

所有任务共享同一套"骨架"：

```
scene(robot + terrain + sensors) 
  + actions(joint_pos) 
  + commands(参考/速度) 
  + observations(policy, critic[, amp_policy, amp_reference]) 
  + rewards(MultiRewardCfg) 
  + terminations/events/curriculum/monitors
```

差异几乎只在**配置层**。这说明作者把"任务 = 配置组合"这件事做到了很高的完成度；新增任务主要是写配置 + 少量 term 函数，而不是改框架。

---

## 7. AMP 专章：与你的 *Actuator Reserve Gating* 的对照

> 你的草稿标题：*Actuator Reserve Gating of Adversarial Motion Priors for Humanoid Locomotion with Calibrated Actuation*。核心是：相位条件判别器 + 接触裕度代理 + 归一化执行器余量（饱和前、扣除预留残差预算）来调制 AMP 风格奖励。

### 7.1 InstinctMJ 的 AMP 实现位置

| 层 | 内容 |
|---|---|
| 算法 | `instinct_rl`：`WasabiPPO`（AMP/WASABI，含判别器、梯度惩罚、logit 权重衰减） |
| 环境接口 | `rl/config.py` 的 `InstinctRlPpoAlgorithmCfg` 暴露 `discriminator_*` 字段；`tasks/parkour/config/g1/rl_cfg.py` 给出具体取值 |
| 状态构造 | `observations/reference_as_state.py` + parkour 配置的 `amp_policy`/`amp_reference` |
| 奖励 | AMP 奖励由算法内部计算，环境侧只提供 task rewards（`MultiRewardCfg`） |

### 7.2 判别器配置逐项解读

| 字段 | 值 | 含义 / 对你的含义 |
|---|---|---|
| `class_name` | `WasabiPPO` | WASABI 风格 AMP |
| `discriminator_kwargs.hidden_sizes` | `[1024, 512]` | 两层 MLP |
| `discriminator_kwargs.nonlinearity` | ReLU | — |
| `discriminator_reward_coef` | `0.25` | 风格奖励权重 |
| `discriminator_reward_type` | `quad` | **二次型奖励**：`clamp(1 - ¼(D-1)², min=0)`，与你草稿中 `r_AMP = max{0, 1 − ¼(D−1)²}` 的形式完全一致（已核对 `instinct_rl/algorithms/wasabi.py` 源码） |
| `discriminator_loss_func` | `MSELoss` | LSGAN 风格：**actor 状态回归 −1，reference 状态回归 +1** |
| `discriminator_gradient_penalty_coef` | `5.0` | 梯度惩罚 |
| `discriminator_logit_weight_decay_coef` | `0.04` | logit 权重衰减 |
| `discriminator_weight_decay_coef` | `3e-4` | 权重衰减 |
| `discriminator_optimizer` | AdamW, lr 1e-4, betas (0.9, 0.999) | — |

**重要发现**：它的 `quad` 奖励公式与你论文中的 AMP 奖励公式**形式相同**。如果你要在论文里写"我们沿用标准二次型 AMP 奖励"，这个仓库可以作为开源实现引用；如果你要主张"相位条件化"是差异点，InstinctMJ 的判别器**没有相位输入**（全仓库没有 gait phase / periodic reward 相关代码，只有执行器延迟的 phase 参数），这会强化你的 novelty 论证。

### 7.3 状态构造方式（可直接借鉴）

parkour 的 `amp_policy` / `amp_reference`：

```
projected_gravity   (3)   × 10 帧 = 30
joint_pos_rel       (29)  × 10     = 290
joint_vel           (29)  × 10     = 290   （scale 0.05）
base_lin_vel        (3)   × 10     = 30
base_ang_vel        (3)   × 10     = 30
                        总计 ≈ 700 维
```

参考侧用同名函数从 `motion_reference` 取（`projected_gravity_reference_as_state` 等），保证"真实状态"与"参考状态"**同一语义、同一维度**——这是 AMP 能对齐的前提。

代码上值得注意的是：`joint_pos_rel_reference_as_state` 用**机器人自身的 default joint pos** 做相对化，而不是参考的 default；这样真机与参考在同一偏移定义下比较。这是一个容易踩坑的细节。

**奖励如何进入 PPO（读源码核对）**：

- 判别器默认键名：`actor_state_key="amp_policy"`、`reference_state_key="amp_reference"`，且 `assert actor_state_size == reference_state_size`；
- 每个 env step，`WasabiPPO.compute_auxiliary_reward()` 用**当前 actor 状态**（detach）过一次判别器，按 `quad` 公式得到风格奖励；
- PPO 基类把该辅助奖励乘以 `discriminator_reward_coef`（=0.25）后**加到环境奖励上**；若环境有多个奖励列，则按 `auxiliary_reward_per_env_reward_coefs` 广播；
- 判别器损失：`MSELoss(actor_d, -1) + MSELoss(reference_d, +1)` 的平均，再加梯度惩罚与 logit 权重衰减。

**对你的启示**：如果你的门控系数 `λ_t` 要乘到风格奖励上，最干净的注入点就是这一步——把标量系数换成逐 env 的张量。但注意：`discriminator_reward_coef` 在 InstinctMJ 里只是配置字段，真正做乘法的是 **instinct_rl 的 PPO 代码**；所以**环境侧无法只靠配置完成门控**，必须改 `instinct_rl/algorithms/ppo.py`（或写一个 `WasabiPPO` 子类）。这是本仓库对你**最有直接工程价值**的一个发现。

### 7.4 与你工作的差异

| 维度 | InstinctMJ | 你的工作 |
|---|---|---|
| 任务 | 复杂地形 parkour（速度/姿态控制） | 人形步态风格 + 执行器不确定性 |
| 判别器输入 | 10 帧本体状态（terrain-conditioned 速度指令在策略侧，不在判别器） | **相位条件**（phase-conditioned） |
| 风格奖励调制 | 固定系数 0.25 | **接触裕度 + 执行器余量门控**（你的核心） |
| 执行器不确定性 | 固定 armature/kp/kd + 延迟；域随机化只有摩擦/质心/推力 | 标定执行器 + 残差扭矩注入（RTI）+ 预留预算 |
| 评测 | 无公开实验数据 | 你要做匹配平台消融（C1–C4） |

### 7.5 可直接复用 / 需要改造的清单

**可直接复用（读代码、抄配置）**

1. `reference_as_state.py` 的参考状态函数族——构造 AMP 参考观测的样板；
2. `motion_reference` 的 mask / multi-frame / lazy update 机制——做"相位窗口"时的数据基础；
3. `assets/unitree_g1.py` 的执行器参数与 kp/kd 反推公式；
4. `monitors/` 的逐项误差与力矩监控字段；
5. `curriculums/motion_reference.py` 的 bin 自适应采样（若你要做参考分布平衡）；
6. `g1_parkour_target_amp_cfg.py` 的奖励/终止/课程组合（作为"训练稳定性"参考）。

**需要改造**

1. 判别器：增加相位输入（InstinctMJ 没有），且相位需要与参考动作对齐（`motion_reference` 的 `time_to_target_frame` 可作为相位来源之一）；
2. 奖励门控：在 `MultiRewardCfg` 之外接入"执行器余量"计算；InstinctMJ 的 reward manager 只接受 term 函数，门控乘子需要作为 term 或包装在算法侧；
3. 执行器不确定性：InstinctMJ 只做固定参数 + 延迟，没有 RTI；你的 RTI 需要自建；
4. 任务：InstinctMJ 的 parkour 是速度跟踪 AMP，不是步态风格 AMP；用它的 G1 平台做步态实验需要重写任务配置与参考数据。

**不能直接对比**

- InstinctMJ 没有公开的训练曲线、成功率、硬件结果；它不是 benchmark；
- 它的任务目标（parkour）与你的（locomotion style）不同；
- 它用 G1，你用 ZOYI Z01（草稿里提到），平台不同；
- 它的 AMP 没有相位条件，不能作为"相位 AMP"的直接 baseline。

---

## 8. 工程质量评估

### 8.1 优点

1. **架构分层干净**：配置（tasks）→ 框架（envs/rl）→ 资产（assets）→ 数据（motion_reference）→ 传感器/地形（sensors/terrains），依赖方向基本单向。
2. **可替换性设计**：entry point、`cfg.class_type`、scene class、reward manager、monitor manager 都可从配置替换；下游仓库能不改源码接自己的实现。
3. **迁移保真度**：大量 `NOTE` 解释"为什么要这样写才能和 IsaacLab 对齐"；提交历史显示作者逐项修复了 actuator delay、AMASS mask、curriculum guard 等语义差异，并声称"14/14 unit tests + 实际构建/步进验证"。
4. **数据管线完整**：PHALP → AMASS → GMR → 过滤 → 元数据配对，全链路脚本化。
5. **感知仿真细致**：深度噪声按真实失效模式建模，含延迟与历史帧。
6. **监控体系**：17 个监控项，覆盖数据集、跟踪误差、力矩、奖励，训练可观测性好。
7. **多卡支持**：`torchrunx` 启动、每 rank 独立 Warp cache、按 rank 切分动作数据。
8. **代码卫生**：775 处返回类型标注、120 处 NOTE、0 处 `except Exception`、pre-commit（black/isort/flake8）。

### 8.2 问题与风险

| 风险 | 说明 | 影响 |
|---|---|---|
| **许可证** | CC BY-NC 4.0，禁止商业使用；继承 IsaacLab 的许可 | 学术可用；商用/产品化需授权；论文引用没问题 |
| **无测试/CI** | `.gitignore` 主动排除 `/tests/` 与 `.github/`，公开快照没有测试 | 无法验证改动；复现问题定位成本高 |
| **无实验产物** | 没有训练日志、曲线、视频、成功率 | 不能作为可复现 benchmark；不能引用性能数字 |
| **数据缺失** | LAFAN1/GMR/AMASS/OMOMO/parkour 数据集、预训练权重都在外部（Google Drive / 本机） | 开箱即用程度低 |
| **路径硬编码** | `~/Xyk/...` 出现 8 处；`/tmp/` 写入 3 处；perceptive VAE 的 `teacher_logdir` 是作者本机路径 | 需要逐处改；`/tmp` 在 Windows 不存在 |
| **导入副作用** | BeyondMimic 配置在 import 时写 `/tmp/{MOTION_NAME}.yaml` | 配置导入即产生副作用，不利于测试与并行 |
| **版本强耦合** | mjlab `1.6.0` editable 同级目录；mujoco-warp 3.11；warp 1.14；Python <3.14 | 升级任一依赖都可能破坏；`uv.lock` 是唯一可靠组合 |
| **单 motion buffer 限制** | 多处 `assert len(motion_buffers) == 1` | 多数据集混合需要改代码 |
| **超大文件** | 1,700 行的 command、1,600 行的 terrain | 维护/阅读成本高 |
| **平台限制** | Linux + NVIDIA GPU；Windows/macOS 训练不可行 | 你的本机无法直接跑 |
| **关节顺序** | README 明确警告：MJCF 关节顺序 ≠ IsaacLab，checkpoint 不通用 | 跨仓库迁移权重会错位 |
| **AMP 在外部库** | 判别器、WASABI 全在 `instinct_rl` | 只读本仓库无法理解 AMP 全貌 |

---

## 9. 对你的论文的具体建议

### 9.1 如何引用/定位这个仓库

- **作为"开源人形 AMP 工程实现"的例证**：说明 AMP 已能在 mjlab/MuJoCo Warp 上以 2048–4096 envs 规模训练，且工程上如何处理执行器延迟、参考状态对齐、多帧历史。
- **不要**把它当作 AMP 算法本身的出处（算法在 instinct_rl / WASABI / AMP 原论文）。
- **不要**把它当作性能 baseline（没有公开指标）。

### 9.2 可写进 related work / implementation 的细节

1. **执行器建模**：`kp = armature·ω²`、`kd = 2ζ·armature·ω`、`ζ=2.0`、`ω=2π·10 Hz`；G1 电机力矩上限 25/88/139/5 N·m；延迟 0–1 步、每回合固定。可作为"标定执行器基线"的对照实现。
2. **AMP 状态构造**：10 帧历史、真实/参考同构、`joint_pos_rel` 用机器人 default 相对化。
3. **二次型 AMP 奖励**：`max(0, 1 - (D-1)²/4)`，与你的公式一致，可作为"标准 AMP 奖励"的实现引用。
4. **训练监控**：逐项跟踪误差 + 力矩 mean/max/min，可作为你的评测指标设计参考。

### 9.3 你的 novelty 相对它仍然成立

- **相位条件判别器**：InstinctMJ 完全没有（grep 无 gait phase / periodic）。
- **执行器余量门控**：InstinctMJ 只有固定执行器参数与延迟，没有"饱和前余量 + 预留预算"的门控。
- **标定执行器 + RTI**：InstinctMJ 没有 RTI；域随机化只有摩擦/质心/推力。
- **匹配平台消融**：InstinctMJ 没有公开消融。

### 9.4 如果要在论文里做"对照实现"

- 最小工作量：把 InstinctMJ 的 `amp_policy`/`amp_reference` 观测构造 + `quad` 奖励 + 执行器参数，移植到你的 Z01 配置里，作为"标准 AMP + 标定执行器"基线（C1/C2 的工程参照）。
- 不建议：直接把它的 G1 parkour 任务搬来对比——任务、平台、评测口径都不同，审稿人会质疑不可比。

---

## 10. 30 分钟阅读路线（按重要性排序）

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `README.md` | 定位、任务列表、安装、警告 |
| 2 | `tasks/parkour/config/g1/rl_cfg.py` | AMP 判别器配置（100 行，最浓缩） |
| 3 | `tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` L540–660 | `amp_policy` / `amp_reference` 构造 |
| 4 | `envs/mdp/observations/reference_as_state.py` | 参考状态函数族 |
| 5 | `motion_reference/motion_reference_manager.py` L1–200 | Sensor 化、lazy update、时间语义 |
| 6 | `motion_reference/motion_buffer.py` | Dataset 抽象（接口设计） |
| 7 | `motion_reference/motion_files/amass_motion.py` L1–220 | 数据加载、Warp 采样、重定向 |
| 8 | `envs/mdp/curriculums/motion_reference.py` | BeyondMimic 自适应采样 |
| 9 | `assets/unitree_g1.py` L100–200 | 执行器参数、延迟语义 |
| 10 | `monitors/monitors.py` L1–120 | 力矩监控 |
| 11 | `sensors/noisy_camera/` + `utils/noise/` | 深度噪声管线 |
| 12 | `terrains/terrain_importer.py` + `virtual_obstacle/edge_cylinder.py` | 地形与虚拟障碍 |

---

## 附录 A：16 个已注册任务 ID

```
Instinct-Locomotion-Flat-G1-v0                  / -Play-v0
Instinct-BeyondMimic-Plane-G1-v0                / -Play-v0
Instinct-Shadowing-WholeBody-Plane-G1-v0        / -Play-v0
Instinct-Perceptive-Shadowing-G1-v0             / -Play-v0
Instinct-Perceptive-Shadowing-G1-OneMotion-v0   / -Play-v0
Instinct-Perceptive-HOI-Shadowing-G1-v0         / -Play-v0
Instinct-Perceptive-Vae-G1-v0                   / -Play-v0
Instinct-Parkour-Target-Amp-G1-v0               / -Play-v0
```

## 附录 B：关键文件索引

| 主题 | 文件 |
|---|---|
| 任务注册 | `src/instinct_mj/tasks/registry.py` |
| Env 扩展 | `src/instinct_mj/envs/manager_based_rl_env.py` |
| 多奖励 | `src/instinct_mj/managers/reward_manager.py` |
| 运动参考 | `src/instinct_mj/motion_reference/motion_reference_manager.py` |
| 数据格式 | `src/instinct_mj/motion_reference/motion_reference_data.py` |
| 数据集加载 | `src/instinct_mj/motion_reference/motion_files/amass_motion.py` |
| 自适应采样 | `src/instinct_mj/envs/mdp/curriculums/motion_reference.py` |
| 命令库 | `src/instinct_mj/envs/mdp/commands/shadowing_command.py` |
| 参考状态观测 | `src/instinct_mj/envs/mdp/observations/reference_as_state.py` |
| 奖励库 | `src/instinct_mj/envs/mdp/rewards/motion_reference.py` |
| 执行器 | `src/instinct_mj/assets/unitree_g1.py` |
| AMP 配置 | `src/instinct_mj/tasks/parkour/config/g1/rl_cfg.py` |
| AMP 环境 | `src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py` |
| 深度相机 | `src/instinct_mj/sensors/noisy_camera/` |
| 深度噪声 | `src/instinct_mj/utils/noise/noise_model.py` |
| 地形 | `src/instinct_mj/terrains/terrain_importer.py` |
| 虚拟障碍 | `src/instinct_mj/terrains/virtual_obstacle/edge_cylinder.py` |
| 监控 | `src/instinct_mj/monitors/monitors.py` |
| 训练入口 | `src/instinct_mj/scripts/instinct_rl/train.py` |
| 播放/ONNX | `src/instinct_mj/scripts/instinct_rl/play.py` |
| VecEnv 适配 | `src/instinct_mj/rl/vecenv_wrapper.py` |

## 附录 C：术语表

| 术语 | 含义 |
|---|---|
| mjlab | MuJoCo Warp 上的 manager-based RL 框架，IsaacLab API 风格 |
| MuJoCo Warp | MuJoCo 的 GPU（CUDA）实现 |
| Motion Reference | 参考运动数据（本项目中作为 Sensor 提供） |
| AMP | Adversarial Motion Priors，用判别器提供风格奖励 |
| WASABI | AMP 的改进算法（本仓库通过 instinct_rl 使用） |
| BeyondMimic | 全身动作跟踪方法，本项目复刻其奖励/采样/执行器建模 |
| Shadowing | 让机器人模仿参考动作（全身/感知/人-物交互） |
| Parkour | 复杂地形运动（速度 + 姿态 + 深度感知 + AMP 风格） |
| VAE Distill | 用 VAE 把教师策略蒸馏为学生策略（去深度依赖） |
| RTI | Residual Torque Injection（你的论文中的训练期残差扭矩注入） |
| CoACD | 凸分解算法，用于把 mesh 地形转成凸碰撞体 |
| Volume Points | 布置在脚/腿体内的点阵，用于检测穿透 |

---

*报告完。基于 `InstinctMJ-main` 快照的静态分析；未运行训练，未验证 GPU 性能与收敛性。*
