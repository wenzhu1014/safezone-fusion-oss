> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# 接驳走廊（Docking Corridor）定义与配置

> 回答博的问题：走廊在哪、怎么判断、能不能自定义。
> 答案：**能自定义**，且支持两种模式——3D 位姿走廊 / 画面区域走廊。

## 1. 走廊是什么 / 为什么重要

走廊（docking_corridor，原 waiting_window）是"小车豁免"的关键门槛之一。
只有 marker **在走廊内**，授权小车的豁免（位置豁免 / 同步运动豁免）才允许生效。
它过滤掉"路过的、歪着的、太远的"小车或物体——只有真正来对接、姿态正确的才进入豁免候选。

豁免三条件（全部满足）：
1. 小车 marker 已授权
2. **marker 在接驳走廊内** ← 本文
3. 物体在 marker 附近的小车本体区域
（人体/肢体永远不豁免）

## 2. 两种走廊模式（config.yaml `docking_corridor.mode`）

### 模式 A：3D 位姿走廊（`mode: pose`，默认）
用小车 marker 的 3D 位姿判断——左右偏移、距离、朝向：
```yaml
docking_corridor:
  mode: pose
  x_min: -0.35        # 左右：柜体正前方左右各 0.35m
  x_max: 0.35
  distance_min: 0.25  # 距离：离柜体 0.25~0.70m
  distance_max: 0.7
  yaw_abs_max_deg: 10 # 朝向：正对柜体 ±10°
```
- **优点**：物理意义明确（小车开到正前方、不远不近、正对着）
- **缺点**：依赖 3D 位姿估计精度。原型阶段（RGB+NIR 无深度）z 距离/yaw 估计有限，需相机标定 + 准确 marker 尺寸才可靠

### 模式 B：画面区域走廊（`mode: image`，推荐俯视固定相机用）
用 marker 在画面里的位置判断——落在画面某块归一化区域内即"在走廊"：
```yaml
docking_corridor:
  mode: image
  x_min: 0.30   # 画面归一化区域（0~1）
  y_min: 0.55
  x_max: 0.70
  y_max: 0.95
camera:
  width: 1280
  height: 720
```
- **优点**：俯视固定相机下"在不在那块地"很直观，**不依赖脆弱的 3D 距离/yaw 估计**，更鲁棒
- **适用**：摄像头固定在柜顶俯视、小车在画面里移动的布局（即本项目）
- marker 中心未知（没看到）→ 判为不在走廊（安全默认，不豁免）

## 3. 怎么选

| 硬件/场景 | 推荐模式 |
|---|---|
| 俯视固定相机（本项目）| **image**（鲁棒，不靠深度）|
| 有可靠深度/双目、需要精确距离门槛 | pose |
| 原型演示、还没标定 | image（先用画面区域，简单可靠）|

两种模式并存，改 `mode` 一个字段切换，其余逻辑（授权、本体区域、人体优先）完全不变。

## 4. 怎么自定义走廊范围

- **pose 模式**：调 `x_min/x_max`（左右）、`distance_min/max`（远近）、`yaw_abs_max_deg`（朝向容差）
- **image 模式**：调归一化 `x_min/y_min/x_max/y_max`（画面里那块区域的位置和大小）
  - 标定方法：让小车停到正确接驳位，看它 marker 在画面里的像素位置，框住那一带即可

## 5. 代码与验证

- `src/geometry.py`：`in_waiting_window`（pose 模式）/ `in_image_corridor`（image 模式）
- `src/safety_zone.py`：`_marker_in_corridor()` 按 mode 分派
- `tests/test_image_corridor.py`（7）：两模式判断、走廊内豁免、走廊外照停、人体永不豁免

## 关键文件
- `src/geometry.py`、`src/safety_zone.py`
- 相关：`COMOTION.md`（同步运动豁免）、`FOLLOWING_ZONE.md`（设备本体排除）
