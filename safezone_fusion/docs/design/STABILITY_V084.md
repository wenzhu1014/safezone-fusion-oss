> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# SafeZone Fusion v0.8.4 停稳判断技术说明

> 目标：判断授权小车是否真正停入柜体接驳等待窗口，并在室内/室外场景下兼顾安全性与抗抖动能力。

## 1. 背景

SafeZone Fusion 的接驳放行依赖三个条件：

1. 授权 marker 可见且 ID 在白名单内；
2. marker 位于等待窗口 / docking corridor；
3. 小车已经停稳，且安全区内没有人员/肢体/未知障碍物。

v0.8.4 主要改进第 3 点：**停稳判断**。

早期版本更多依赖瞬时速度、平滑速度或中心点抖动判断。但真实摄像头场景下会出现：

- ArUco 角点重定位导致瞬时速度尖峰；
- 手持/户外风扰造成原地小幅晃动；
- 真实小车缓慢移动时，单帧速度可能不高；
- 摄像头距离变化会改变像素位移尺度。

因此 v0.8.4 新增 `anchored_position` 模式：

> 不只问“这一帧速度快不快”，而是问“最近一段时间 marker 有没有一直待在锚点范围内”。

---

## 2. 核心模式：anchored_position

配置：

```yaml
waiting_window:
  stability_mode: anchored_position
  position_hold_time_sec: 1.0
  position_anchor_drift_max_px: 8.0
  position_history_radius_max_px: 3.0
  position_center_speed_warn_px_s: 5.0
```

### 2.1 判断条件

`anchored_position` 同时检查：

| 条件 | 含义 |
|---|---|
| 同一个 marker 连续可见 | 避免偶发识别误判 |
| `anchor_drift <= position_anchor_drift_max_px` | 当前中心点没有离开初始锚点太远 |
| `center_jitter <= position_history_radius_max_px` | 最近窗口内中心点没有大范围发散 |
| `anchor_age >= position_hold_time_sec` | 已经在锚点附近保持足够时间 |

满足后才输出 stable。

### 2.2 调试 reason

稳定时 reason 类似：

```text
cart_in_window_and_stable:anchored_position frames=... drift=...px radius=...px hold=...s center_speed=...px/s note=...
```

其中：

- `drift`：当前点距离 anchor 的像素距离；
- `radius`：近期中心点半径 / 抖动幅度；
- `hold`：在 anchor 附近保持的时间；
- `note=calm`：速度也较平稳；
- `note=fast_local_sway`：瞬时速度较高，但仍在锚点范围内，更像原地晃动。

---

## 3. 与 angled_marker 的关系

`angled_marker` 是 v0.8.3 的抗斜角/抗风扰模式，逻辑是：

- 连续看到同一个授权 marker；
- 没有大幅 anchor 漂移；
- 低速保持一段时间；
- 如遇原地小晃动，可通过 `wind_sway_*` 容忍。

`anchored_position` 更进一步，把“位置历史”作为主判断。

建议：

- 保留 `angled_marker` 作为回退和对照；
- 默认候选使用 `anchored_position`；
- 实测时可通过命令行切换对比。

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --stability-mode angled_marker
python demo_live_camera.py --camera-index 0 --marker-id 123 --stability-mode anchored_position
```

---

## 4. 室内/室外 profile

v0.8.4 增加环境 profile：

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment indoor
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment outdoor
```

`--environment` 会先应用 profile；`--stability-mode` 可在 profile 后覆盖具体模式，方便调试。

### 4.1 indoor

适用：室内、柜体固定、摄像头结构稳定、风扰很小。

```yaml
environment_profiles:
  indoor:
    waiting_window:
      stability_mode: anchored_position
      position_hold_time_sec: 0.8
      position_anchor_drift_max_px: 6.0
      position_history_radius_max_px: 2.5
      position_center_speed_warn_px_s: 5.0
      wind_sway_anchor_drift_max_px: 6.0
      wind_sway_jitter_max_px: 2.5
```

特点：

- 更敏感；
- 更快确认停稳；
- 更容易拒绝细微移动；
- 适合室内稳定安装。

### 4.2 outdoor

适用：户外、轻微风扰、柜体/支架可能小幅晃动。

```yaml
environment_profiles:
  outdoor:
    waiting_window:
      stability_mode: anchored_position
      position_hold_time_sec: 1.0
      position_anchor_drift_max_px: 8.0
      position_history_radius_max_px: 3.0
      position_center_speed_warn_px_s: 5.0
      wind_sway_anchor_drift_max_px: 8.0
      wind_sway_jitter_max_px: 3.0
```

特点：

- 稍慢确认停稳；
- 容忍原地小晃动；
- 仍拒绝持续离开 anchor 的移动；
- 比单纯风速传感器更直接，因为它看的是 marker/柜体实际相对位移。

---

## 5. 标定流程

### 5.1 基础停稳标定

```bash
python tools/calibrate_marker_stability.py --camera-index 0 --marker-id 123
```

阶段：

1. `static`：marker 固定不动；
2. `slow_move`：模拟小车缓慢移动。

输出：

```text
calibration_runs/marker_stability_*/samples.csv
calibration_runs/marker_stability_*/summary.json
```

### 5.2 户外/风扰标定

```bash
python tools/calibrate_marker_stability.py --camera-index 0 --marker-id 123 --sway-sec 5
```

阶段：

1. `static`：完全静止；
2. `sway`：原地轻晃 / 模拟风扰；
3. `slow_move`：慢速平移。

`summary.json` 会包含：

```json
"sway_recommendation": {
  "wind_sway_anchor_drift_max_px": ...,
  "wind_sway_jitter_max_px": ...,
  "wind_sway_min_anchor_age_sec": ...
}
```

注意：脚本推荐值是参考，不应无脑采用最大值。若推荐的 anchor 范围过大，可能把慢速移动也容忍成风扰。

当前经验值：

```yaml
wind_sway_anchor_drift_max_px: 8.0
wind_sway_jitter_max_px: 3.0
wind_sway_min_anchor_age_sec: 0.3
```

---

## 6. 是否需要风速传感器

现阶段不建议把风速传感器作为必需项。

原因：

- 风速不是直接安全变量；
- 风大不一定导致 marker 相对位移大；
- 风小也可能因结构松动造成明显抖动；
- 视觉锚点/位置历史能直接反映真实影响。

更推荐的优先级：

1. 视觉锚点 + `anchored_position`；
2. 柜体参考 marker / 背景稳像；
3. IMU / 简单振动传感器；
4. 风速传感器作为运维/预警辅助。

风速传感器适合后续这些场景：

- 商业部署需要记录天气/强风导致的暂停；
- 强风时直接切入保守策略；
- 户外误报明显，且需要提前预警而不是只靠视觉结果。

---

## 7. 当前推荐默认策略

### 原型阶段

保留全局默认：

```yaml
waiting_window:
  stability_mode: angled_marker
```

通过命令行试运行 profile：

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment indoor
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment outdoor
```

### 若 anchored_position 持续表现稳定

可考虑将默认切到：

```yaml
waiting_window:
  stability_mode: anchored_position
```

并根据部署环境选择：

- 室内：`indoor` profile；
- 室外：`outdoor` profile。

---

## 8. 实测检查表

完整实测步骤见 [`ACCEPTANCE_V084.md`](ACCEPTANCE_V084.md)。本节仅保留摘要。

每次调整参数后至少验证：

- [ ] marker 固定不动：能在预期时间内 stable；
- [ ] marker 原地轻晃：不应频繁误判为移动；
- [ ] marker 平移：应立刻或短时间内回到等待；
- [ ] marker 消失/遮挡：应重置 anchor；
- [ ] 大幅跳点：不应误放行；
- [ ] 室内/室外 profile 行为差异符合预期。

---

## 9. 关键文件

- `src/waiting_window.py`：停稳模式核心逻辑；
- `src/vision/aruco_detector.py`：marker 检测、center、jitter、speed 估计；
- `config.yaml`：waiting_window 参数与 environment_profiles；
- `demo_live_camera.py`：实时摄像头 demo，支持 `--environment` 和 `--stability-mode`；
- `tools/calibrate_marker_stability.py`：static / sway / slow_move 标定工具；
- `tests/test_waiting_window_stability.py`：停稳逻辑单元测试。
