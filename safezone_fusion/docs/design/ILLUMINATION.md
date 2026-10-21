> Historical prototype design note. Numeric thresholds are examples.
> Validation claims describe the original note, not this public candidate; see the root VALIDATION.md.
> This project has no demonstrated safety certification.

# 光照鲁棒性与阴影抑制（室内外通用）

> 起源：博 2026-06-22 问"还可能遇到哪些问题"。光照变化和阴影是室内外（尤其户外）
> 占用检测最高频的误报源。本特性让系统对它们鲁棒，同时**不牺牲安全**。

## 1. 解决两个高频误报

### 全局光照变化
一片云飘过、灯切换、黄昏、车灯扫过 → 整个画面较均匀地变亮/变暗。
原始背景差分会"到处亮起来" → 误触发 STOP。
**解法**：估计 roi 与背景之间的**均匀亮度偏移**（取中位数，鲁棒），差分前补偿掉，
只有**局部变化**才留下来。补偿有上限（`max_global_shift`），避免真实大变化被完全抵消。

### 投射阴影
人/云的影子扫过柜面 = 一片移动的暗区 → 经典误报。
**解法**：识别"变暗但是均匀比例变暗"的像素（阴影 ratio 带）→ 从障碍 mask 里剔除。

## 2. ⚠️ 安全命门：深色物体 ≠ 阴影（必须靠纹理区分）

**只靠亮度无法区分阴影和深色物体**——穿深色衣服的人，亮度可能只有背景的 40~60%，
和阴影一模一样。**如果只看亮度，会把深色衣服的人当阴影抑制掉 → 漏检 → 危险！**

（开发时实测确认了这个隐患：ratio 0.4~0.6 的物体会被误判为阴影。）

**真正的区别是纹理**：
- 阴影：**保留**底层表面纹理（地面纹理透过阴影还在，只是整体变暗）→ 可抑制
- 实体物体：**替换**了表面纹理（盖住地面，换成自己的表面）→ 必须保留（STOP）

所以阴影判定加了**纹理保留门**（`require_texture_preserved`）：
一个像素被判为阴影，必须同时满足 ①在阴影 ratio 带 **且** ②局部纹理与背景一致。
存疑时**保留像素**（fail toward STOP，安全偏置）。

## 3. 配置（config.yaml `roi_motion.illumination`）

```yaml
roi_motion:
  illumination:
    enabled: true
    compensate_global: true     # 补偿全局亮度变化
    max_global_shift: 60.0      # 补偿上限，防真实大变化被抵消
    suppress_shadow: true       # 阴影抑制
    shadow_ratio_low: 0.35
    shadow_ratio_high: 0.92
    shadow_min_darken: 8.0
    require_texture_preserved: true  # ★ 安全：深色物体≠阴影，靠纹理区分
    texture_flat_thresh: 6.0
    texture_keep_ratio: 0.45
```

### 室内 / 室外建议
- **室内**：光照较稳，可把 `compensate_global` 留开但影响小；阴影抑制按需
- **户外（带挡雨板）**：挡雨板已挡掉直接雨水和顶部强光，难点收敛为"光照变化+阴影"——
  本特性正是为此；保持 `require_texture_preserved: true` 确保深色物体不漏检

## 4. 验证

`tests/test_illumination.py`（10）：
- 全局偏移估计 / 补偿后均匀变亮→diff≈0 / 局部物体仍突出
- 阴影像素被识别 / 极暗遮挡物不算阴影
- **深色衣服物体（纹理被替换，纹理化背景上）→ 不被当阴影**（安全）
- **阴影（纹理保留）→ 被抑制**
- 集成：全局变亮→不触发；真实物体→照触发；阴影→不触发

全套件：**186 passed**。

## 5. 安全声明

- 本特性**只抑制清晰的光照/阴影**，存疑保留 → fail toward STOP
- 纹理保留门确保深色物体不被误抑制（漏检是安全大忌）
- 与分层检测协同：语义手检测层独立运行，不受本特性影响

## 关键文件
- `src/vision/illumination.py`、`src/vision/roi_motion_detector.py`
- 相关：`COMOTION.md`、`FOLLOWING_ZONE.md`、`DOCKING_CORRIDOR.md`
