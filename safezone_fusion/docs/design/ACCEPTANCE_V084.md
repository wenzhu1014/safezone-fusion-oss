> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# SafeZone Fusion v0.8.4 实测验收清单

> 目的：把“看起来效果不错”转化为可复测、可交付的验收记录。  
> 范围：ArUco marker 授权小车进入接驳等待窗口后的停稳判断，重点覆盖 `anchored_position`、室内/室外 profile、风扰/晃动、遮挡与异常情况。

## 0. 测试前准备

### 0.1 环境

- [ ] 摄像头可被 OpenCV 打开；
- [ ] ArUco marker ID 与命令行 `--marker-id` 一致；
- [ ] marker 尺寸、打印质量、光照条件基本稳定；
- [ ] 测试时没有其他程序占用摄像头；
- [ ] 当前代码测试通过：

```bash
python -m pytest tests\test_waiting_window_stability.py
python -m py_compile demo_live_camera.py src\waiting_window.py tools\calibrate_marker_stability.py
```

### 0.2 建议记录字段

每个场景记录：

```text
日期/时间：
环境：indoor / outdoor
摄像头 index：
marker id：
stability_mode：
观察到的 reason：
是否通过：PASS / FAIL
备注：
```

重点看 overlay / 终端中的：

- `decision`
- `motor`
- `reason`
- `center_v`
- `drift`
- `radius`
- `hold`
- `note=calm / note=fast_local_sway`
- `sway=yes`

---

## 1. 基础命令

### 1.1 室内模式

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment indoor
```

预期：更敏感，停稳确认较快，细微平移更容易被拒绝。

### 1.2 室外模式

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment outdoor
```

预期：稍微容忍原地小晃动，但持续平移仍应等待/不放行。

### 1.3 模式对比

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --stability-mode angled_marker
python demo_live_camera.py --camera-index 0 --marker-id 123 --stability-mode anchored_position
```

预期：`anchored_position` 应更符合“在一段时间内没有离开锚点范围”的直觉。

---

## 2. 室内 profile 验收

### IN-01 固定 marker 停稳

**操作：**

1. 运行 `--environment indoor`；
2. 将 marker 固定在等待窗口内；
3. 保持不动 2 秒。

**预期：**

- `decision` 进入允许小车/接驳相关状态；
- reason 包含：

```text
cart_in_window_and_stable:anchored_position
```

- `hold >= 0.8s`；
- `drift <= 6px`；
- `radius <= 2.5px`。

**通过标准：** 连续 3 次测试均可在约 0.8~1.5 秒内稳定。

---

### IN-02 室内轻微原地晃动

**操作：**

1. marker 保持在原地附近；
2. 轻微晃动但不平移离开原位置。

**预期：**

- 若 `drift <= 6px` 且 `radius <= 2.5px`，可保持 stable；
- 若晃动超过室内阈值，应回到等待/不稳定。

**通过标准：** 小范围晃动不频繁误拒，大范围晃动会被拒绝。

---

### IN-03 室内平移

**操作：**

1. marker 从等待窗口内慢慢横向平移；
2. 平移距离明显超过原锚点范围。

**预期：**

- reason 出现类似：

```text
cart_not_stable:position_anchor_drift=...px>6.0px
```

- 不应继续保持 stable。

**通过标准：** 平移超过阈值后能及时拒绝。

---

## 3. 室外 profile 验收

### OUT-01 固定 marker 停稳

**操作：**

1. 运行 `--environment outdoor`；
2. marker 固定在等待窗口内；
3. 保持不动 2 秒。

**预期：**

- reason 包含 `cart_in_window_and_stable:anchored_position`；
- `hold >= 1.0s`；
- `drift <= 8px`；
- `radius <= 3px`。

**通过标准：** 连续 3 次测试均可稳定。

---

### OUT-02 户外/风扰原地轻晃

**操作：**

1. 使用 `--environment outdoor`；
2. 原地轻晃 marker 或支架，模拟风扰；
3. 不要刻意平移离开锚点。

**预期：**

- 小幅晃动可保持 stable；
- reason 可能出现：

```text
note=fast_local_sway
```

或在 angled_marker 对比模式中出现：

```text
sway=yes
```

**通过标准：** 原地小晃动不应频繁误判移动。

---

### OUT-03 户外 profile 平移

**操作：**

1. 使用 `--environment outdoor`；
2. marker 连续平移，模拟小车未停稳。

**预期：**

- 超过 `position_anchor_drift_max_px: 8.0` 后拒绝；
- 不应因 outdoor 容忍而继续放行。

**通过标准：** 平移超过 8px 后稳定状态被取消。

---

## 4. 异常与安全场景

### SAFE-01 marker 遮挡 / 消失

**操作：**

1. marker 已 stable；
2. 用手遮挡 marker 或移出画面。

**预期：**

- 系统不应继续保持小车 stable；
- anchor 应重置或等待重新建立。

**通过标准：** marker 消失后不能继续放行。

---

### SAFE-02 marker ID 错误

**操作：**

1. 使用非白名单 marker；
2. 或命令行指定错误 `--marker-id`。

**预期：**

- 不授权；
- 不应进入允许小车状态。

**通过标准：** 错 ID 不放行。

---

### SAFE-03 大幅跳点 / 快速移动

**操作：**

1. 快速移动 marker；
2. 或让 marker 突然跳到另一个位置。

**预期：**

- reason 出现 anchor drift / radius 超限；
- 不应误判 stable。

**通过标准：** 大幅跳点不放行。

---

### SAFE-04 人/肢体进入 stop_zone

**操作：**

> 当前 demo 若没有真实人体检测，可先用 mock/后续真实传感器补测。

1. 小车 marker stable；
2. 人或手进入 stop_zone。

**预期：**

- 人/肢体永远不因授权 marker 豁免；
- 输出 STOP 或对应安全停止。

**通过标准：** 人体/肢体进入即停。

---

## 5. 摄像头与距离场景

### CAM-01 摄像头距离变化

**操作：**

1. 在近距离、中距离、预计部署距离分别测试；
2. 分别执行固定、轻晃、平移。

**预期：**

- 距离越远，像素位移越小；
- 若远距离下平移不易触发，需要重新标定 anchor/radius 参数。

**通过标准：** 预计部署距离下，平移仍能被拒绝，固定仍能稳定。

---

### CAM-02 光照变化

**操作：**

1. 明亮、偏暗、背光、反光场景分别测试；
2. 观察检测稳定性和 rejected 候选数量。

**预期：**

- 光照变化不应导致频繁误 stable；
- 检测失败时应进入等待，而不是保持旧 stable。

**通过标准：** 检测不稳定时系统保守等待。

---

### CAM-03 单摄 / 双摄兼容

**操作：**

单摄：

```bash
python demo_live_camera.py --camera-index 0 --marker-id 123 --environment indoor
```

双摄预留：

```bash
python demo_live_camera.py --camera-index 0 --secondary-index 1 --primary-role auto --secondary-role auto --marker-id 123 --environment indoor
```

**预期：**

- 单摄可正常运行；
- 第二路接入后不破坏单摄逻辑；
- 双摄后续可用于深度/背景/柜体参考。

**通过标准：** single-first / dual-ready 流程可用。

---

## 6. 标定验收

### CAL-01 基础标定

```bash
python tools\calibrate_marker_stability.py --camera-index 0 --marker-id 123
```

**预期：**

- 生成 `calibration_runs/marker_stability_*/samples.csv`；
- 生成 `summary.json`；
- static 阶段 detected > 90%。

**通过标准：** 能输出有效统计，不在 marker 缺失时给误导阈值。

---

### CAL-02 sway 标定

```bash
python tools\calibrate_marker_stability.py --camera-index 0 --marker-id 123 --sway-sec 5
```

**预期：**

- summary 包含 `sway` 和 `sway_recommendation`；
- 推荐值仅作为参考，不能盲目使用过大的 anchor/radius。

**通过标准：** 能区分 static / sway / slow_move 三段数据。

---

## 7. 通过标准总结

v0.8.4 可视为通过验收，当满足：

- [ ] 室内固定可稳定；
- [ ] 室内平移可拒绝；
- [ ] 室外原地轻晃可容忍；
- [ ] 室外平移可拒绝；
- [ ] marker 消失不继续放行；
- [ ] 错误 ID 不放行；
- [ ] 预计部署距离下仍能区分固定/平移；
- [ ] 标定脚本能生成有效数据；
- [ ] 单元测试通过。

---

## 8. 当前推荐结论

若实测通过，推荐将部署默认策略设为：

```yaml
waiting_window:
  stability_mode: anchored_position
```

并按场景选择：

```bash
--environment indoor
# 或
--environment outdoor
```

如果 outdoor 模式把明显平移误判为稳定，应收紧：

```yaml
position_anchor_drift_max_px: 6.0
position_history_radius_max_px: 2.5
```

如果 outdoor 模式对原地风扰过于敏感，可小幅放宽：

```yaml
position_anchor_drift_max_px: 9.0
position_history_radius_max_px: 3.5
```

不建议一次性放宽过大；每次调整后重新执行 IN/OUT/SAFE 三类核心测试。

---

## 9. 合成验收测试（无新硬件）

为了在第二根摄像头线到位前验证完整决策链路，已新增合成验收测试：

```bash
python -m pytest tests\test_acceptance_v084_synthetic.py -q
```

覆盖：

- indoor 固定 marker → `ALLOW_CART`；
- indoor 平移离开 anchor → `WARNING`；
- outdoor 原地轻晃 → `ALLOW_CART` 且 `note=fast_local_sway`；
- marker 缺失 → `IDLE`；
- marker ID 错误 → `IDLE`；
- human/hand 进入 stop zone → `STOP`；
- 未知障碍物进入 stop zone 且不属于授权小车本体 → `STOP`；
- 物体进入 warning zone → `WARNING`；
- 授权小车本体对象可豁免，但人体/肢体永远不豁免。

当前验证：

```text
tests/test_acceptance_v084_synthetic.py: 9 passed
tests/test_waiting_window_stability.py + synthetic acceptance: 17 passed
```
