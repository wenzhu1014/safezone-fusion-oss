from __future__ import annotations

"""Turn a diagnostic recording into a human-readable "why did it stop" report.

On-site usage::

    python tools/diag_summary.py out/diag.jsonl

Prints:
  * overall counts (frames, stop ratio, fps min/mean, stop reasons breakdown)
  * the stop/start transition timeline (the thing an operator actually asks about)

No third-party deps.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from safezone_fusion.src.diagnostics import load_diag, summarize, stop_transitions

# Friendly Chinese labels for common reasons.
_REASON_CN = {
    "limb_in_stop_zone": "肢体进入停止区",
    "human_in_stop_zone": "人体进入停止区",
    "unknown_obstacle_in_stop_zone": "未知障碍物进入停止区",
    "lens_blinded": "镜头被遮挡/致盲",
    "unknown": "未知",
}

def _cn(reason: str) -> str:
    if not reason:
        return "(无)"
    for key, label in _REASON_CN.items():
        if key in reason:
            return f"{label} [{reason}]"
    return reason

def main(argv) -> int:
    if len(argv) < 2:
        print("用法: python tools/diag_summary.py <diag.jsonl>")
        return 2
    path = argv[1]
    rows = load_diag(path)
    if not rows:
        print(f"[空] {path} 没有可分析的记录")
        return 1

    s = summarize(rows)
    print("=" * 56)
    print(f"诊断报告: {path}")
    print("=" * 56)
    print(f"总帧数      : {s['frames']}")
    print(f"停止帧数    : {s['stop_frames']}  (占比 {s['stop_ratio']*100:.1f}%)")
    print(f"帧率        : 均值 {s['fps_mean']} / 最低 {s['fps_min']}")
    print(f"停止切换次数: {s['transitions']}")
    print("-" * 56)
    print("停止原因分布:")
    if s["stop_reasons"]:
        for reason, n in sorted(s["stop_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {_cn(reason)}")
    else:
        print("  (整段录制没有停止)")
    print("-" * 56)
    print("停止/放行 时间线 (每次状态翻转):")
    trans = stop_transitions(rows)
    if not trans:
        print("  (无翻转)")
    else:
        t0 = trans[0]["ts"]
        for t in trans:
            rel = t["ts"] - t0 if t.get("ts") else 0.0
            state = "■ STOP " if t["stop"] else "□ 放行 "
            print(f"  +{rel:7.2f}s  帧{t['frame']:<6} {state} <- {_cn(t.get('reason') or '')}")
    print("=" * 56)
    # Health hint
    if s["fps_min"] and s["fps_min"] < 12:
        print("⚠️ 最低帧率低于 12fps —— 安全距离可能不足，需排查算力/降负载。")
    if s["stop_ratio"] > 0.8:
        print("⚠️ 停止占比过高 —— 可能存在持续误触发，检查光照/阴影/ROI 设置。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
