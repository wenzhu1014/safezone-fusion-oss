# SafeZone Fusion 接口文件 v0.7

> 本文件用于实时同步算法与外部系统的接口。后续每次接口变化都要更新这里。

## 总体数据流

```text
外卖柜业务系统 / 调度系统
  -> BusinessInput

视觉与安全传感器
  -> CameraPacket / PirPacket
  -> VisionSensorFusion
  -> SensorFrame

SafeZoneOrchestrator.tick(RuntimeInput)
  -> RuntimeOutput

电机 / 门锁 / 语音执行层
  <- ActuatorOutput
```

---

## 坐标系约定

所有位置默认使用柜体坐标系，单位米：

```text
x: 左右方向，柜体中心为 0，右正/左负可按现场约定，但必须全系统一致
y: 高度方向，地面向上为正
z: 距离柜体方向，柜体向外为正
```

角度：

```text
yaw_deg: 小车/marker 相对柜体正面的偏航角，单位度
```

---

## OperationMode

```python
IDLE           # 空闲
CART_DOCKING   # 小车接驳模式
HUMAN_ACCESS   # 人工取餐/存餐模式
```

---

## BusinessInput

来源：外卖柜业务系统 / 调度系统。

```python
BusinessInput:
  mode: OperationMode
  slot_request: SlotRequest | None
  docking_requested: bool
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| mode | OperationMode | 当前业务模式 |
| slot_request | SlotRequest/null | 人工取餐/存餐模式下的目标格口请求 |
| docking_requested | bool | 小车接驳流程是否正在请求 |

---

## SlotRequest

来源：外卖柜业务系统 + 设备当前位置反馈。

```python
SlotRequest:
  slot_id: str
  slot_x: float
  slot_y: float
  head_x: float
  head_y: float
  door_closed: bool
  user_done: bool
```

### 字段说明

| 字段 | 单位 | 说明 |
|---|---:|---|
| slot_id | - | 目标格口 ID |
| slot_x | m | 目标格口中心 x |
| slot_y | m | 目标格口中心 y |
| head_x | m | 当前移动接驳头中心 x |
| head_y | m | 当前移动接驳头中心 y |
| door_closed | bool | 当前目标格口门是否关闭 |
| user_done | bool | 用户是否完成取餐/存餐 |

---

## Vision 前置接口

v0.5 新增视觉前置接口，见：

```text
src/vision/interfaces.py
```

### CameraPacket

来源：视觉/深度传感器。

当前电脑原型：一路 RGB 彩色 + 一路黑白/NIR。  
最终柜体部署：预计接入双目或深度相机，因此接口保留 stereo/depth/disparity/point_cloud 字段。

```python
CameraPacket:
  rgb: Any
  nir: Any
  mono: Any
  left_rgb: Any
  right_rgb: Any
  left_depth: Any
  right_depth: Any
  depth: Any
  disparity: Any
  point_cloud: Any
  timestamp: float
  meta: dict
```

说明：不强绑定 OpenCV/Numpy，所以图像/深度字段暂为 `Any`。真实接入时可以是 numpy array、SDK frame、点云对象或图像路径。

阶段策略：
- 当前原型：`rgb + nir/mono + pir`，以 2D ROI、marker 几何估距、人体/肢体检测、轮廓/运动冗余为主。
- 最终柜体：`stereo/depth/point_cloud + rgb + pir`，以三维动态安全包络判断为主。

### PirPacket

来源：PIR / 毫米波雷达 / 人体感应传感器。

```python
PirPacket:
  triggered: bool
  timestamp: float
  meta: dict
```

### MarkerDetector

```python
detect(packet: CameraPacket) -> MarkerObservation | None
```

当前真实原型：`src/vision/aruco_detector.py` 中的 `OpenCVArucoMarkerDetector`。

当前默认 marker 配置：

```text
dictionary: DICT_6X6_250
marker_id: 123
marker_size_m: 0.10（原型默认）
```

注意：OpenCV `DICT_6X6_250` 的 ID 范围是 0-249；示例 ID 使用 123，避免超出字典范围。

后续真实实现可扩展为 AprilTag / QR 识别。

### ObjectDetector

```python
detect(packet: CameraPacket) -> list[ObjectObservation]
```

后续真实实现：双目深度点云聚类 + YOLO/pose/hand detector。

### PirReader

```python
read() -> PirPacket
```

后续真实实现：GPIO/串口/Modbus/传感器 SDK。

### VisionSensorFusion

```python
read(packet: CameraPacket) -> SensorFrame
```

负责将 marker、人体/障碍物、人体感应融合为 `SensorFrame`。

---

## SensorFrame

来源：VisionSensorFusion。

```python
SensorFrame:
  marker: MarkerObservation | None
  objects: list[ObjectObservation]
  pir_triggered: bool
  timestamp: float
```

---

## MarkerObservation

来源：ArUco / AprilTag / QR 视觉识别模块。

```python
MarkerObservation:
  marker_id: int
  pose: Pose
  stable_frames: int
  speed_mps: float
  visible: bool
```

### 字段说明

| 字段 | 单位 | 说明 |
|---|---:|---|
| marker_id | - | 小车 marker ID |
| pose | Pose | marker 在柜体坐标系下的位置/姿态 |
| stable_frames | 帧 | 连续稳定识别帧数 |
| speed_mps | m/s | marker/小车估计速度 |
| visible | bool | marker 当前是否可见 |

---

## ObjectObservation

来源：人体检测、肢体检测、双目点云聚类、障碍物检测。

```python
ObjectObservation:
  label: str
  pose: Pose
  confidence: float
  belongs_to_verified_cart: bool
```

### label 推荐值

```text
person, human, hand, arm, leg, foot, head, body, limb, unknown
```

### 说明

`belongs_to_verified_cart` 通常由安全融合层根据授权 marker 与空间关系自动标记，真实传感器层不必直接提供。

---

## Pose

```python
Pose:
  x: float
  y: float
  z: float
  yaw_deg: float = 0.0
```

---

## RuntimeInput

统一输入。

```python
RuntimeInput:
  business: BusinessInput
  sensors: SensorFrame
  tick_id: int
```

---

## RuntimeOutput

统一输出。

```python
RuntimeOutput:
  mode: OperationMode
  actuator: ActuatorOutput
  fusion: FusionResult
```

---

## ActuatorOutput

给真实执行层的抽象命令。

```python
ActuatorOutput:
  motor: MotorAction
  door: DoorAction
  voice: str | None
  reason: str
  raw_decision: str | None
  debug: dict
```

---

## MotorAction

```python
HOLD                 # 保持不动
STOP                 # 立即停止/急停/软停，按硬件策略实现
ALLOW_CART_DOCKING   # 授权小车已满足条件，允许进入接驳后续流程
MOVE_TO_PARK         # 移动设备到避让位
MOVE_TO_STANDBY      # 移动设备回待机位
LOCK_STILL           # 设备静止锁定，允许用户靠近
```

---

## DoorAction

```python
NONE         # 不改变门状态
OPEN_SLOT    # 打开目标格口
KEEP_CLOSED  # 保持关闭
```

---

## 当前已验证 demo

```bash
python projects/safezone_fusion/demo.py
python projects/safezone_fusion/demo_human_access.py
python projects/safezone_fusion/demo_orchestrator.py
python projects/safezone_fusion/demo_runtime_loop.py
python projects/safezone_fusion/demo_vision_pipeline.py
python projects/safezone_fusion/demo_aruco_image.py
```

---

## 待接入 adapter

| Adapter | 输入/输出 | 说明 |
|---|---|---|
| BusinessAdapter | 输出 BusinessInput | 接外卖柜业务系统、调度系统、用户输入 |
| VisionSensorFusion | CameraPacket/PirPacket -> SensorFrame | 接双目摄像头、人体识别、PIR/雷达 |
| SensorAdapter | 输出 SensorFrame | 旧称，可由 VisionSensorFusion 替代或封装 |
| ActuatorAdapter | 输入 ActuatorOutput | 控制电机、门锁、语音 |

---

## 安全原则

1. 设备运动前必须确认 stop_zone 清空。
2. 设备运动中，人/肢体进入 stop_zone 必须 STOP。
3. 设备静止锁定时，允许用户靠近取餐/存餐。
4. 授权小车只豁免小车本体/接驳走廊，不豁免人和未知障碍物。
5. 不依赖柜体实时远程控制小车微调；小车应自主停入接驳等待窗口。
