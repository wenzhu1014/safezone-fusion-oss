> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# 跟随式动态危险区（设备移动 + 本体排除）

> 回答两个核心问题：
> ① 在柜面上移动的接驳设备会不会被当成障碍物？→ 不会，本体被排除。
> ② 检测障碍物的框能不能自定义？→ 能，而且它会跟着设备自动移动。

## 1. 设备形态与摄像头布局（前提）

- 设备：市面外卖柜 + 柜顶固定摄像头（俯视整个柜面）
- 柜面上有一个"背包"状接驳设备，可在**整个柜面上下左右移动**
- 摄像头**固定**，设备在画面里移动
- **PLC 知道设备当前位置**（它驱动电机，自然知道 X/Y）

## 2. 为什么不能画一个静态大框

设备**能**走遍整个柜面，但任何**瞬间**只占一小块。
若画个大框覆盖它所有可能位置 → 框 = 整个柜面 → 安全检测形同虚设。

**夹手风险只发生在"正在移动的设备本体周围"**，柜面远处没有机构、没有夹的风险，不需要保护。

## 3. 正确方案：危险区跟着设备走

```
PLC 设备位姿 (x,y)  ──映射像素──>  危险区 = 设备周围一圈
                                  ├─ 设备本体那块：排除（那是它自己）
                                  └─ 本体外一圈：检测人手/肢体侵入
设备移动 → 这个小框跟着移动。永远只看设备的移动邻域，不看整个柜面。
```

- **危险区（danger_box）**：设备本体 + 四周 margin，是"要检测人手"的区域
- **设备本体框（device_box）**：从危险区里**抠掉**，本体在里面怎么动都不算障碍物
- 设备**移动时**危险区会额外膨胀一点（`inflate_when_moving`），给运动机构留余量

## 4. 代码组成

| 文件 | 作用 |
|---|---|
| `src/following_zone.py` | 纯几何：设备位姿 → danger_box + device_box（归一化）；含 `FaceToPixelCalibration` 归一化→像素映射 |
| `src/interfaces.py` `DevicePose` | PLC 设备位姿输入（x,y 归一化 0..1，moving 标志）|
| `src/vision/roi_motion_detector.py` | `set_dynamic_roi()` 设跟随框、`set_device_mask_px()` 设本体排除框 |

### 关键 API
```python
# 每帧从 PLC 位姿算跟随区
zone = compute_following_zone(pose.x, pose.y, pose.moving, zcfg)
detector.set_dynamic_roi(zone.danger_box)            # 检测框跟随设备
dev_px = calib.box_to_pixels(zone.device_box, w, h)
detector.set_device_mask_px(dev_px)                  # 排除设备本体
```

## 5. 关于"框能不能自定义"——三个层次

1. **静态自定义**（最简单）：改 `config.yaml` 的 `roi_motion.roi: [x0,y0,x1,y1]`（归一化 0~1），框就变。
2. **跟随设备**（本特性）：传 PLC 位姿，框自动跟着设备移动，无需手动画。
3. **形状参数可调**：`FollowingZoneConfig` 里 `device_half_w/h`（本体大小）、`margin`（危险环宽度）、`inflate_when_moving`（移动时膨胀）、`device_mask_pad_px`（本体遮罩 padding）都可调。

## 6. 现场要做的唯一标定：位姿→像素映射

PLC 给的是设备坐标（毫米或柜面归一化），摄像头看的是像素，中间要标定一次：

1. 驱动设备到柜面**四个角**（或若干已知点）
2. 记录每个点设备在画面里的**像素中心**
3. 拟合 `FaceToPixelCalibration`（fronto-parallel 俯视相机用简单 scale+offset 即可）

标定前可用 `FaceToPixelCalibration.identity(w,h)` 作起步（假设柜面铺满画面）。

## 7. 安全兜底

- **PLC 位姿丢失/未知时**：`whole_face_when_unknown=True`（默认）→ 退回看一个保守的默认框，**不会因为"不知道设备在哪"就停止检测**（安全偏置）。
- **人体/肢体永远最高优先级**：本体排除只对"设备自己"生效；任何人手/肢体进入危险环照样 STOP。
- **ROI 大幅移动会重置背景**：避免用旧背景误判新区域。

## 8. 验证

- `tests/test_following_zone.py`：跟随/小框/本体排除/移动膨胀/边缘钳制/标定映射（9）
- `tests/test_following_zone_integration.py`：**设备本体在遮罩内移动不触发 STOP；手进危险环照样 STOP**（4）

## 关键文件
- `src/following_zone.py`、`src/vision/roi_motion_detector.py`、`src/interfaces.py`（DevicePose）
- 部署对接：`DEPLOYMENT.md`
