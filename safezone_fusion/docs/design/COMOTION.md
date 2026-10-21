> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# 同步运动豁免（Co-Motion Exemption）

> 起源：博 2026-06-22 的想法——"小车肯定和码同步，能不能检测和码同步运动的物体"。
> 这比"在 marker 周围圈一块豁免区"更聪明：用**运动关联**而不是**位置**来区分车身和闯入物。

## 1. 核心思想

小车和它身上的 marker 是**刚性连接**的——码往哪走、车身就往哪走，速度方向一致。
而一只从旁边伸进来的手，**不会跟着码同步运动**。

所以：
```
随 marker 同步运动的那团  → 刚性连接 → 车身 → 豁免
运动矢量和 marker 不一致  → 独立闯入 → STOP
```

这是计算机视觉里的 **common fate / 刚体运动分组**原理。

## 2. 为什么比"位置圈地"好

| | 位置圈地（marker 附近一圈豁免）| 同步运动（本方案）|
|---|---|---|
| 判断依据 | 在不在框里 | **跟不跟码一起动** |
| 手混在车身附近 | 可能误豁免 ❌ | 手的运动≠码的运动，抓得出 ✅ |
| 小车形状不规则/超框 | 遮不全 ❌ | 跟码走的都算，不靠预设大小 ✅ |

## 3. 关键安全设计（豁免不能放水）

同步运动豁免**只在以下全部成立时**才生效：
1. **marker 已授权 + 在接驳走廊内**（`exempt_active=True`，调用方判定）——运动关联本身永不单独豁免
2. **marker 确实在移动**（速度 > `min_speed_px`）——停着没有运动矢量可比
3. **运动块与 marker 同向**（角差 ≤ `max_angle_deg`，默认 25°）**且同速**（速度比在 `speed_ratio_min~max`，默认 0.5~2.0）

并且：
- **人体/肢体永远不参与豁免**——语义手检测层（mediapipe_hand）独立运行、永不被共动豁免，分层防御不破
- **小车停稳时**（marker 速度≈0）→ 共动**判不了**（decidable=False）→ 不靠共动豁免，**交回位置豁免**（marker 附近区域，由语义层 `safety_zone.py` 处理已停稳车身）
- marker 丢失 → 速度追踪 coasting 几帧后归零 → 豁免自动撤销

## 4. 动 / 静两套豁免协同

| 小车状态 | 豁免依据 | 由谁处理 |
|---|---|---|
| **移动接近中** | 同步运动（本方案）| `comotion.py` + `roi_motion_detector` |
| **停稳对接中** | marker 附近区域（位置）| `safety_zone.py` verified_cart_region |
| 人体/肢体 | 永不豁免 | 全层最高优先级 STOP |

## 5. 代码组成

| 文件 | 作用 |
|---|---|
| `src/comotion.py` | 纯向量逻辑：`MotionTracker`（质心→速度，带 coasting）+ `is_co_moving()`（同向同速判定）|
| `src/vision/roi_motion_detector.py` | `set_marker_comotion(centroid_px, exempt_active)`；在触发判定前做共动豁免 |

### 调用方每帧要做的
```python
# marker 授权且在走廊内才允许共动豁免
exempt = cart.verified and marker_in_corridor
detector.set_marker_comotion(marker_centroid_px, exempt_active=exempt)
```

## 6. 验证

- `tests/test_comotion.py`（9）：速度追踪/coasting；同向同速→共动；异向/异速/反向→不共动；停稳→不可判
- `tests/test_comotion_integration.py`（4）：
  - 车身随码同步滑入 → **豁免不停**
  - 手从异向伸入（码在动）→ **照停**
  - marker 未授权 → **不豁免**
  - 小车停稳 → 共动不可判 → **不被共动误放**

## 7. 现场调参

`CoMotionConfig`：
- `min_speed_px`：低于此速度认为 marker 停了（默认 1.5 px/帧）——按实际帧率/小车速度标定
- `max_angle_deg`：同向容差（默认 25°）
- `speed_ratio_min/max`：同速容差带（默认 0.5~2.0）

## 8. ⚠️ 已界定的边界（阴影面）

**任何「靠运动关联」的方法都有一个固有盲区**：一只手如果和小车**完全同速同向**移动，
纯靠运动分不出它是手还是车体（`test_comotion_boundary.py` 已明确钉下这个事实）。

**但不会造成危险，因为有三层兜底：**
1. **语义手检测层独立、永不被共动豁免** —— mediapipe 一旦认出「手」，不管跟不跟车同步，照样停。共动豁免只作用于「说不清是什么」的运动块，永远无法把「手」变成「放行」。
2. **现实中手伸进去的方向不一样** —— 人够柜子是朝柜体方向伸，小车是横向接近，矢量对不上。「完美同步」需要人精确跟车同速平移，现实难自然发生。
3. **参数可收紧** —— 调严 `max_angle_deg`/`speed_ratio` 可进一步缩小对抗窗口。

安全工程原则：**知道并界定自己的盲区，比假装没有盲区更可信。**

## 关键文件
- `src/comotion.py`、`src/vision/roi_motion_detector.py`
- 相关：`FOLLOWING_ZONE.md`（设备本体排除）、`safety_zone.py`（停稳车身位置豁免）
