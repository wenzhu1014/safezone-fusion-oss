> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# 运行状态门控：启动安全 + 帧率监控

> 起源：博 2026-06-22 排查"还可能遇到哪些问题"。这是两个安全相关、纯软件、便宜的加固。

## 1. 解决两个真问题

### 启动盲区
系统刚上电时，背景模型还没学好、检测器还没热身。这几秒**不能信任它能看见危险**
——"还没就绪" ≠ "安全无人"。所以**启动期默认 STOP**，等 warmup 完成才允许运动。

### 帧率退化
最小安全距离（ISO 13855）是按某个**假设 FPS** 算的。如果摄像头板 CPU 卡顿、FPS 掉了，
真实响应时间变长，**配置的安全距离悄悄变得不够了**。所以实时测量 FPS，
低于地板值就触发 fail-safe STOP。

## 2. 行为

每帧 `tick()` 返回 `stop_required`，调用方 OR 进 STOP 决策（和镜头自检同一 fail-safe 模式）：

- **warmup 期**：`ready=False` → `stop_required=True`（需同时满足帧数 + 时间两个条件）
- **FPS 退化**：连续 `degrade_after` 帧低于 `min_fps` → 锁存 degraded → `stop_required=True`
- **恢复**：FPS 需回升到 `recover_fps` 以上并持续 `recover_after` 帧才解除（迟滞防抖）

## 3. 配置（config.yaml `operating_state`）

```yaml
operating_state:
  enabled: true
  warmup_frames: 15         # 离开启动 STOP 门控所需帧数
  warmup_min_seconds: 0.5   # 同时需要的最短墙钟时间
  expected_fps: 30.0        # 安全距离按此 FPS 计算
  min_fps: 12.0             # 持续低于此 -> 退化 -> fail-safe STOP
  recover_fps: 15.0         # 回升超过此才解除（迟滞）
  fps_window: 30
  degrade_after: 5
  recover_after: 10
```

## 4. 现场注意

- `min_fps` 必须和 `RESPONSE_TIME.md` 里算安全距离用的 FPS 一致——**两者要对齐**
- 若目标摄像头板实测帧率偏低，应：①重算安全距离用更低 FPS，或 ②优化检测降负载
- warmup 帧数按背景学习需要的时间设（默认 15 帧 + 0.5s 适合 ~30fps）

## 5. 验证

`tests/test_operating_state.py`（7）：
- 启动期 STOP 直到 warmup 完成
- warmup 需同时满足帧数 + 时间
- 健康 FPS 不退化 / 低 FPS 连续退化后锁存
- 退化后迟滞恢复
- 禁用时恒就绪 / reset 回到 warmup

全套件：**193 passed**。

## 关键文件
- `src/operating_state.py`、`demo_live_camera.py`（主循环接入）
- 相关：`RESPONSE_TIME.md`（FPS 与安全距离）、`lens_health.py`（同 fail-safe 模式）
