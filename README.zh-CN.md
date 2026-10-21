# SafeZone Fusion

用于教学、仿真与集成研究的 Python 视觉接驳决策原型。包含 ArUco 标记识别、
停稳判定、分层检测接口、危险区融合、镜头健康监测、运行状态门控和信号看门狗。

从仓库根目录运行，已验证的 Python 版本为 3.11：

```sh
python -m pip install -r requirements-dev.txt
python -m safezone_fusion.demo
python -m pytest safezone_fusion/tests -q
```

建议先创建并激活独立虚拟环境。默认 Demo 使用合成观察数据，不打开摄像头。
默认测试不运行需要原始录像的 `slow` 套件；录像不随项目发布。

`safezone_fusion/config.yaml` 只是演示配置。视觉标记不等于安全身份认证，
运动检测不等于人体识别。可选 MediaPipe 检测器需要自行提供兼容环境与本地模型；
本项目不自动下载模型。本次没有验证真实模型推理或硬件控制。

本项目没有已证明的安全认证，测试通过不能据此认定可安全控制真实机械。
项目角色、来源和公开整理范围见 [PROVENANCE.md](PROVENANCE.md)，
验证边界见 [VALIDATION.md](VALIDATION.md)，完整说明见 [README.md](README.md)。
