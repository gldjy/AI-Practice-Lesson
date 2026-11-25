#!/usr/bin/env python
"""
Shared joint metadata for the arm viewer, server and client.

Keep the ranges in sync with the JOINTS table in arm_model.html.
"""

JOINTS = [
    {"id": 1, "name": "夹爪", "min": 500, "max": 665, "initial": 500},
    {"id": 2, "name": "腕部旋转", "min": 145, "max": 500, "initial": 500},
    {"id": 3, "name": "腕部俯仰", "min": 120, "max": 500, "initial": 500},
    {"id": 4, "name": "小臂", "min": 500, "max": 860, "initial": 500},
    {"id": 5, "name": "大臂", "min": 150, "max": 740, "initial": 500},
    {"id": 6, "name": "底座", "min": 140, "max": 860, "initial": 500},
]

LIMITS = {item["id"]: (item["min"], item["max"]) for item in JOINTS}
DEFAULT_POSE = {item["id"]: item["initial"] for item in JOINTS}


def normalize_pose(raw):
    """Validates and normalizes a complete {joint: angle} mapping."""
    if not isinstance(raw, dict):
        raise ValueError("pose 必须是对象")

    pose = {}
    for key, value in raw.items():
        if isinstance(key, bool) or isinstance(value, bool):
            raise ValueError(f"无效的角度: {key}={value}")
        try:
            joint = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"无效的关节: {key}")
        if str(joint) != str(key).strip():
            raise ValueError(f"无效的关节: {key}")
        if joint not in LIMITS:
            raise ValueError(f"未知关节: {joint}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"无效的角度: {joint}={value}")
        angle = int(value)
        if angle != value:
            raise ValueError(f"角度必须是整数: {joint}={value}")
        low, high = LIMITS[joint]
        if not low <= angle <= high:
            raise ValueError(f"{joint} 号超出范围 {low}~{high}: {angle}")
        pose[joint] = angle

    missing = sorted(set(LIMITS) - set(pose))
    if missing:
        raise ValueError(f"缺少关节: {missing}")
    return {joint: pose[joint] for joint in sorted(pose)}


def format_pose(pose):
    """Formats a pose as a compact string."""
    return "  ".join(f"{joint}:{pose[joint]}" for joint in sorted(pose))
