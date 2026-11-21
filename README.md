# Z01 v1.6 · PartyMimic

基于 PartyMimic 的 **Z01 v1.6 全身动作跟踪项目**：包含机器人模型、G1 → Z01 动作重定向、轨迹约束与筛选、策略训练、仿真评估和播放。机器人名称统一为 `z01`，包含 **26 个转动关节、27 个连杆**，支持 `joint` 和 `sparse5` 参考接口。

本项目仅保留 Z01 运行链路。G1 模型与 LAFAN1 数据作为重定向输入和动作对照保留；训练和播放入口仅支持 Z01。

## 目录

```text
Z01_v1.6_whole_body/         原始 Z01 v1.6 模型：MJCF、URDF、USD、STL
assets/
  z01/                     训练模型 robot.xml、控制参数 control.json
  hhtools_robots/{g1,z01}/  重定向所需的源/目标机器人预设和标定
src/partymimic/             训练、仿真、运动学、策略与世界模型
  z01/                     Z01 关节、跟踪连杆、镜像及驱动配置
scripts/train.sh           训练启动脚本
tools/                    数据处理、模型检查、评估、播放和 ONNX 工具
tests/                    轨迹约束、动作采样、速度惩罚及数据接口测试
docs/z01.md               模型、标定、筛选阈值与已知限制
data/                     本地下载与生成数据，不纳入 Git
logs/                     Z01 训练权重、日志和审计产物，不纳入 Git
```

## 环境与模型检查

需要 Linux x86_64、Python 3.10–3.13、[uv](https://docs.astral.sh/uv/)；训练与策略仿真需要 NVIDIA GPU。锁文件使用 PyTorch 2.9.0 / CUDA 12.8 和 Warp 1.16.0，应安装兼容的 NVIDIA 驱动。交互播放需要桌面显示；离线视频需要 EGL 和 `ffmpeg`。

在项目根目录运行：

```bash
uv sync --locked
uv run python tools/data/z01_data.py prepare
uv run python tools/check_z01.py
uv run python -m unittest discover -s tests -v
```

`prepare` 从原始模型生成训练 MJCF 和重定向 URDF，保留既有标定文件。`check_z01.py` 检查 26 关节顺序、驱动参数、正向运动学、镜像，以及 5 ms 仿真步长下的 5 秒无支撑站立。

修改 `assets/z01/control.json` 后，应重新执行生成和检查。控制参数是仿真初始值；模型惯量、上半身安装偏移等约束详见[技术说明](docs/z01.md)。

## 动作数据

若本地已有 `data/motions/z01_lafan_v2_clean/*.npz`，可直接检查参考动作并开始训练。重新构建时按以下顺序执行。

### 1. 准备重定向源

```bash
bash tools/download_data.sh
bash tools/data/prepare_source.sh
```

只下载 G1 的 LAFAN1 CSV 和所需几何，生成源 MJCF 与 50 Hz NPZ：

- `data/lafan1_g1/`：30 Hz 原始 CSV。
- `data/g1_description/`：源 URDF、网格和用于对照的 MJCF。
- `data/motions/g1/lafan/`：50 Hz 源动作，仅用于 Z01 重定向和审计。

### 2. G1 → Z01 重定向与筛选

另行准备 [human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools)（本地流程使用提交 `ed8b350`，在其目录运行 `uv sync --extra all`）。默认在当前项目的同级目录寻找该工具，也可以指定 `HHT`：

```bash
HHT=/path/to/human-humanoid-tools P=4 bash tools/data/build_z01.sh
```

流程为：G1 NPZ → HHT 输入 CSV → Newton IK → Z01 NPZ → 关节角度/速度约束、地面修正 → 逐帧筛选。输出目录：

| 路径 | 用途 |
|---|---|
| `data/raw/z01_v2/` | 输入 CSV、重定向 CSV、工作进程日志、输入哈希清单 |
| `data/motions/z01_lafan_v2_ik/` | 尚未约束速度和地面的 IK 结果 |
| `data/motions/z01_lafan_v2/` | 修正后的完整动作、分段与 `conditioning.json` |
| `data/motions/z01_lafan_v2_clean/` | 筛选后的相对链接，以及 `selection.json` |

现有本地筛选报告记录：40 个源动作、441,140 帧；保留来自 37 个源动作的 163 段，共 233,935 帧，约 1.299 小时。具体样本以 `selection.json` 为准，重新构建后应重新审计。

短流程检查应使用独立目录：

```bash
HHT=/path/to/human-humanoid-tools P=1 \
  Z01_RAW=data/raw/z01_check Z01_OUT=data/motions/z01_check \
  bash tools/data/build_z01.sh --clip walk2_subject3 --limit-frames 1000
```

脚本仅在输入文件、标定、HHT 提交和参数匹配时复用 CSV。输入改变时使用新的 `Z01_RAW` / `Z01_OUT`，保留原实验记录。对复制项目，动作链接使用相对路径；历史 JSON 报告中的绝对路径保留为当时的来源记录。

### 3. 审计与参考播放

```bash
uv run python tools/audit_z01_motions.py \
  --motions data/motions/z01_lafan_v2 \
  --source data/motions/g1/lafan --out logs/z01_audit

uv run python tools/play_reference.py \
  --motion data/motions/z01_lafan_v2_clean/walk2_subject3__seg000.npz \
  --speed 0.5 --loop
```

参考播放直接设置姿态，不运行策略或物理积分。空格暂停，R 重播，鼠标控制视角。离线视频示例：

```bash
uv run python tools/play_reference.py \
  --motion data/motions/z01_lafan_v2_clean/walk2_subject3__seg000.npz \
  --seconds 10 --video logs/z01_reference.mp4
```

## 训练

所有入口默认使用 `z01`；默认动作目录为 `data/motions/z01_lafan_v2_clean`。以下是 16 GB 显存的起始配置，建议先完成 200 次迭代检查，确认显存、数值和速度指标后再进行完整训练：

```bash
bash scripts/train.sh logs/z01_check \
  --num_envs 1024 --buffer_steps 500 --pi_batch 512 --wm_batch 1024 \
  --v_gcons_n 1024 --joint_velocity_penalty 0.5 --joint_velocity_margin 0.9 \
  --iterations 200

bash scripts/train.sh logs/z01_train \
  --num_envs 1024 --buffer_steps 500 --pi_batch 512 --wm_batch 1024 \
  --v_gcons_n 1024 --joint_velocity_penalty 0.5 --joint_velocity_margin 0.9 \
  --iterations 12000 --seed 0
```

输出包括 `args.json`、`stdout.log`、`train_log.csv`、周期权重 `ckpt_*.pt` 和 `last.pt`。查看全部选项：`uv run python -m partymimic.train -h`。

- `--motion_dir <目录>`：使用另一个 Z01 NPZ 动作集。
- `--resume <last.pt>`：继续训练；`--init <checkpoint>`：用已有 Z01 策略初始化新实验。
- `--interface sparse5`：使用骨盆、头部、双腕和双踝参考；`--ref_enc xattn`：切换参考编码器。
- `--joint_velocity_penalty 0.5 --joint_velocity_margin 0.9`：添加超过 URDF 声明速度 90% 的训练惩罚，**不提供硬速度限制**。该惩罚的程序默认值为 0，需显式传入。

`train_log.csv` 中 `velocity_over_frac` 表示采样状态中任一关节超过 URDF 速度的比例，`velocity_ratio_max` 表示该记录区间的最大速度比。

## 策略播放与评估

```bash
uv run python tools/play.py --ckpt logs/z01_train/ckpt_012000.pt \
  --motion data/motions/z01_lafan_v2_clean/walk2_subject3__seg000.npz --loop

# 无窗口的有限步数检查
uv run python tools/play.py --ckpt logs/z01_check/last.pt \
  --motion data/motions/z01_lafan_v2_clean/walk2_subject3__seg000.npz \
  --headless --steps 100

uv run python -m partymimic.evaluate --run_dir logs/z01_train \
  --ckpt logs/z01_train/ckpt_012000.pt --out logs/z01_train/evaluation.json
```

策略播放使用训练仿真器运行物理和控制，机器人类型从权重读取。评估逐个动作从首帧运行，关闭随机化与观测噪声。批量评估入口为 `tools/eval_batch.py`；记录的轨迹可用 `tools/render.py` 渲染。

可选的单动作 ONNX 导出与回放工具为 `tools/export_onnx.py`、`tools/play_onnx.py`，仅支持 `joint` 接口。先运行 `uv sync --locked --extra onnx`，使用 `--help` 查看参数。ONNX 导出不等于完成实机控制集成。

本地 `logs/z01_*` 和旧版 Z01 动作集保留用于实验追溯。旧的 `z01_lafan` / `z01_lafan_clean` 存在手臂标定与筛选问题，新实验使用 `z01_lafan_v2_clean`。已有权重是否达到任务要求需要独立评估；短训练日志和资产检查不能证明策略收敛。

## 数据格式与许可

每个 Z01 `.npz` 至少包含 `fps=50`、`joint_pos [T,26]`、`body_pos_w [T,B,3]` 和 `body_quat_w [T,B,4]`。第一个 body 是骨盆，四元数顺序为 **wxyz**，角度单位为弧度；关节顺序由 `src/partymimic/z01/z01_constants.py` 定义，其他连杆状态通过正向运动学计算。

代码继承 PartyMimic 及其上游的许可，见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。[LAFAN1 G1 重定向数据](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)来自第三方；LAFAN1 标注为 CC BY-NC-ND 4.0，Unitree G1 描述为 BSD-3-Clause。动作数据及其衍生产物不因存放在本项目中而自动适用代码的 MIT 许可。Z01 原始模型保留本地提供的来源说明，分发时另行确认其授权。
