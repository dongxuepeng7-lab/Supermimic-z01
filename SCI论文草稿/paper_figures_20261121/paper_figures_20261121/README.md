# 论文图片：已有辨识数据与独立回放结果

生成日期：2026-11-21。图片英文标注，提供 300 dpi PNG、矢量 PDF、可编辑 SVG。源脚本 `make_figures.py` 可重现全部图片；`source_hashes.json` 记录输入文件 SHA256。

## 这些图片能支持什么

展示已有记录的异步命令/反馈时序、已有参数辨识的独立预测效果，以及限位失配和踝部耦合瞬态。没有新增采集、重新拟合或改动模型/控制参数。

这些是**参数集合对比**，不是“连续PD / 固定周期PD / 实际事件PD”三种辨识方法的公平对比。不能使用“proposed method outperforms all baselines”或“real-world cross-frequency generalization verified”等表述。事件机制的创新性、参数补偿减小和跨频率优势仍待专门消融与已知参数实验支持。

原始模型指在对应冻结CAD上恢复目标关节原始参数，不代表整机历史版本。汇总图包含本次整理的8条2026-11-20独立验证记录，并非所有关节或全部历史验证。各关节幅值、周期、PD、轨迹长度不同；仅进行同一关节同一记录内的比较，不对关节间改善幅度进行显著性排名。

## 图1：已有数据中的异步时序证据

文件：`fig01_recorded_timing.pdf` / `.png` / `.svg`

四个面板分别展示固定50ms窗口的命令/反馈事件、完整回放区间的事件间隔经验分布、命令到达时最近一次严格早于命令的反馈年龄，以及整段时间间隔变化。来源：right_ankle_pitch_scale0.3_T3。窗口从回放开始后3s取起，不根据抖动程度挑选。

这里的“反馈年龄”是bag记录时间差，额外命令延迟取0，与本次右踝候选一致；同时间戳采用命令先于反馈的约定。它不是传感器实际采样年龄，也不能解释为独立测得的通信延迟。图中不同频率不是多个控制频率工况实验。

**English caption:** Recorded command and feedback timing for the held-out right ankle pitch trajectory (scale 0.3, period 3 s). (a) Event timestamps in a fixed 50-ms window. (b) Empirical distributions of inter-event intervals over the complete replay span. (c) Age of the latest strictly earlier recorded feedback at each command event, assuming zero additional command delay. (d) Inter-event intervals throughout the replay span. Timestamps describe recorded receipt events rather than verified hardware execution times.

## 图2：独立验证误差比

文件：`fig02_holdout_error_ratios.pdf` / `.png` / `.svg`

位置、速度、力矩三项指标均以目标关节原始模型RMSE归一化；虚线1表示原始模型，低于1为改善，高于1为退化。8组完整记录包含左右膝、左右髋俯仰、左髋侧摆、右髋旋转、右踝俯仰和左踝侧摆。左踝侧摆失败结果保留。

采用0.5ms最大积分步长结果，插值到对应原2ms评价时刻，使用原实测数组及评分掩码。首段预热排除规则沿用原报告，所有后续瞬态均保留。每个数值均从数组重新计算并核对已有报告。无独立重复试验，因此没有伪造置信区间或把时间样本当作独立实验。

髋/膝/左踝侧摆用已有已部署舍入参数；右髋旋转、右踝俯仰用冻结精确候选，具体版本记录在 `figure_metrics.csv`。右踝候选未部署。右踝相对“原始右侧模型”的速度改善不等同于相对“左侧复用参数”的速度改善，后者在此前报告中退化约5.46%。

**English caption:** Held-out trajectory prediction errors normalized by the corresponding original-parameter model. Results use a maximum integration step of 0.5 ms and the original common 2-ms scoring grid. Position, velocity, and torque errors are evaluated over each complete scored trajectory, including transients. Dashed lines denote parity; ratios above one indicate degradation. Labels specify the recording scale and period, not a common physical amplitude. The comparison evaluates parameter sets and does not isolate the effect of asynchronous timing modeling.

## 图3：代表性独立轨迹

文件：`fig03_heldout_trajectories.pdf` / `.png` / `.svg`

左右列依次为左膝、左髋侧摆和右踝俯仰；三行是位置、力矩和位置残差。原始模型、辨识集合与实测相比较。统一选择评分开始后的前三个周期，不选择最好看的时间窗口，不重新初始化、不移相、不平滑轨迹。整段统计见图2。

**English caption:** Held-out trajectories for the left knee, left hip roll, and right ankle pitch. Measured signals are compared with continuous predictions from the original and identified parameter sets. The panels show position, joint-side torque, and position residuals. Each window contains the first three scored periods of an existing full-span rollout; no state reset, phase adjustment, or additional fitting is performed for visualization.

## 图4：必须报告的适用边界

文件：`fig04_failure_modes.pdf` / `.png` / `.svg`

左上展示左踝侧摆实测越过XML下限-0.25rad而仿真受限的现象；左下展示保留限位及仅离线关闭目标限位时的完整记录位置RMSE，两种情况下辨识集合均不优于原始模型。关闭限位仅为已有离线诊断，不是部署建议。

右侧展示右踝侧摆偏离中位数最大的瞬态附近位置与关节侧力矩；roll命令保持0，但状态发生剧烈变化。窗口由最大绝对roll偏离确定，不作一般性能窗口。用户已确认采集时观察到明显抖动；图不能确定抖动根因。单关节模型固定同侧其他自由度，无法复现完整并联踝耦合。

**English caption:** Limitations of the scalar joint model. (a) Left ankle roll measurements extend beyond the XML lower joint limit. (b) Removing this limit only in an offline diagnostic reduces absolute error but does not make the identified parameter set outperform the original set. Gray and blue bars denote original and identified sets, respectively. (c–d) Measured right ankle pitch/roll motion and joint-side feedback torque around the largest roll excursion, despite a constant roll command. This transient motivates coupled-ankle modeling but does not identify its cause.

## 图5：参数清单

文件：`fig05_parameter_inventory.pdf` / `.png` / `.svg`

灰色为2026-11-20已有运行参数汇总快照，不声称是全部独立辨识真值或最新实时配置。蓝色菱形为右踝俯仰新舍入候选，未写入运行模型。负载偏置L对应仿真外力矩-L，延迟单位ms。零惯量/阻尼/延迟是模型/拟合取值，不表示硬件没有惯量、阻尼或通信时延。右踝侧摆复用左侧配置，左踝侧摆未通过跨幅值验证；右髋侧摆历史留出验证亦未改善，不应以“已部署”代替“已验证”。

**English caption:** Parameter inventory from the archived runtime summary dated 20 November 2026, with the subsequently evaluated right ankle pitch candidate overlaid as blue diamonds. Load bias L corresponds to an external torque of -L. Runtime deployment does not imply independent identification or successful validation of every joint. Zero-valued fitted parameters should not be interpreted as absence of the corresponding physical effects.

## 复现

从工程根目录执行：

```sh
.venv/bin/python analysis/paper_figures_20261121/make_figures.py
```

`checks.json` 记录校验口径；`figure_metrics.csv` 中q单位rad、v单位rad/s、tau单位N·m。时序统计保存于 `timing_statistics.json`。归档的历史输入文件和报告未改写。
