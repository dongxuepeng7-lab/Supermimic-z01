# Z01 v1.6 训练资产

- `robot.xml`：由原始 MJCF 生成的训练模型。
- `control.json`：仿真控制参数和站姿；修改后需重新生成模型。

在项目根目录运行：

```bash
uv run python tools/data/z01_data.py prepare
uv run python tools/check_z01.py
```

安装、动作准备、训练与播放见[项目 README](../../README.md)。模型、标定与筛选限制见[技术说明](../../docs/z01.md)。原始模型见 [Z01_v1.6_whole_body](../../Z01_v1.6_whole_body/README.md)。
