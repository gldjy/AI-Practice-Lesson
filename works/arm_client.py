#!/usr/bin/env python
"""
Mirrors the pose published by arm_server.py to the physical arm.

Run on the kit:
    python arm_client.py --server 127.0.0.1 --port 8000 --serial /dev/ttyUSB0

Test the network side without hardware:
    python arm_client.py --server 127.0.0.1 --dry-run
"""

import argparse
import json
import threading
import time
from urllib.parse import urlparse
from urllib.request import urlopen

from arm_params import DEFAULT_POSE, format_pose, normalize_pose

APPLY_LOG_INTERVAL = 0.5
last_apply_log = 0.0


def normalize_base(server, port):
    """Turns a host or URL from --server into a http(s) base URL."""
    if "://" not in server:
        server = "http://" + server
    parsed = urlparse(server)
    if not parsed.hostname:
        raise ValueError(f"无效的服务器地址: {server}")
    scheme = parsed.scheme or "http"
    actual_port = parsed.port or port
    return f"{scheme}://{parsed.hostname}:{actual_port}"


def fetch_pose(base, timeout=2.0):
    """Reads and validates the current pose from the server."""
    with urlopen(base + "/api/pose", timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        payload = json.load(response)
    raw = payload.get("pose") if isinstance(payload, dict) and "pose" in payload else payload
    return normalize_pose(raw)


def wait_for_pose(base):
    """Blocks until the server answers with a valid pose."""
    while True:
        try:
            pose = fetch_pose(base)
        except Exception as exc:
            print(f"\r[Info] 等待服务器 {base} ({exc})", end="", flush=True)
            time.sleep(2.0)
            continue
        print(f"\r[Info] 已连接 {base}        ")
        return pose


def heartbeat(uart):
    """Sends the system heartbeat until the uart stops."""
    while uart.running:
        uart.transmitSysHeart()
        time.sleep(0.2)


def apply_pose(uart, pose, previous=None):
    """Sends only the joints that differ from the previous pose."""
    global last_apply_log

    changed = [
        joint
        for joint in sorted(pose)
        if previous is None or pose[joint] != previous.get(joint)
    ]
    for joint in changed:
        uart.setArmAngle(joint, pose[joint])
        time.sleep(0.02)
    if changed:
        now = time.time()
        if now - last_apply_log >= APPLY_LOG_INTERVAL:
            print(f"[Apply] {format_pose(pose)}  changed={changed}")
            last_apply_log = now
    return changed


def apply_pose_group(uart, pose, wait_ms=1500):
    """Sends the whole pose as one coordinated action group."""
    row = tuple(pose[joint] for joint in range(1, 7)) + (wait_ms,)
    uart.setArmActions((row,))
    deadline = time.time() + wait_ms / 1000.0 + 3.0
    while uart.getArmActionsBusy() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(1.5)


def wait_action(uart, timeout=10.0):
    """Waits until the last action group should have finished."""
    deadline = time.time() + timeout
    while uart.getArmActionsBusy() and time.time() < deadline:
        time.sleep(0.05)


def vendor_stand_pose(uart):
    """Returns the final angles of the vendor stand action group."""
    frames = [frame for frame in uart.robotArm.actionStand if len(frame) > 6]
    return {joint: frames[-1][joint - 1] for joint in range(1, 7)}


GROUP_DELTA = 100


def needs_group(previous, pose):
    """Returns True for multi-joint or large single-joint moves."""
    if previous is None:
        return True
    changed = [
        joint for joint in sorted(pose) if pose[joint] != previous.get(joint)
    ]
    if not changed:
        return False
    return len(changed) > 1 or max(
        abs(pose[joint] - previous[joint]) for joint in changed
    ) >= GROUP_DELTA


def large_deltas(pose):
    """Returns joints that are far from the safe stand pose."""
    return {
        joint: pose[joint] - DEFAULT_POSE[joint]
        for joint in sorted(pose)
        if abs(pose[joint] - DEFAULT_POSE[joint]) >= GROUP_DELTA
    }


def run_dry(base, rate, last):
    """Prints pose changes without opening the serial port."""
    delay = 1.0 / rate
    while True:
        try:
            pose = fetch_pose(base)
        except Exception as exc:
            print(f"[Warning] 服务器不可用: {exc}")
            time.sleep(1.0)
            continue
        if pose != last:
            print(f"[Pose] {format_pose(pose)}")
            last = pose
        time.sleep(delay)


def run_arm(args, base, target_pose):
    """Applies server poses to the arm until interrupted."""
    try:
        from uart import Uart
    except ImportError as exc:
        print("[Error] 无法导入 uart（需要 pyserial），请先安装:", exc)
        return 1

    uart = Uart(args.serial)
    try:
        uart.start()
    except Exception as exc:
        print(f"[Error] 无法打开串口 {args.serial}: {exc}")
        return 1

    try:
        threading.Thread(target=heartbeat, args=(uart,), daemon=True).start()
        time.sleep(1.0)  # Let the lower controller finish booting.
        uart.setEnable(True)
        time.sleep(0.5)

        print("[Info] 执行厂家 stand 初始化...")
        uart.robotArmMoveit(uart.robotArm.Movement.stand)
        wait_action(uart)
        time.sleep(3.0)  # Same settle time as arm_test.py.
        stand_pose = vendor_stand_pose(uart)
        print(f"[Info] 厂家 stand: {format_pose(stand_pose)}")
        last = stand_pose

        print(f"[Info] 调整到 server 姿态: {format_pose(target_pose)}")
        apply_pose(uart, target_pose, stand_pose)
        last = target_pose
        lost = False
        delay = 1.0 / args.rate

        while True:
            try:
                pose = fetch_pose(base)
                if lost:
                    print("[Info] 服务器已恢复")
                    lost = False
            except Exception as exc:
                if not lost:
                    print(f"[Warning] 服务器不可用: {exc}")
                    lost = True
                time.sleep(1.0)
                continue

            if pose != last:
                if needs_group(last, pose):
                    print("[Info] 多关节或大角度变化，使用动作组")
                    apply_pose_group(uart, pose)
                else:
                    apply_pose(uart, pose, last)
                last = pose
            time.sleep(delay)
    except Exception as exc:
        print(f"[Error] 控制循环异常: {exc}")
        return 1
    finally:
        print("[Info] 复位机械臂并关闭串口...")
        try:
            uart.stop(0, 0)
        except Exception as exc:
            print(f"[Warning] 关闭串口时出错: {exc}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="把 arm_server.py 的姿态同步到实体机械臂")
    parser.add_argument("--server", default="127.0.0.1", help="服务器主机或完整地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器未指定端口时使用")
    parser.add_argument("--serial", default="/dev/ttyUSB0", help="机械臂串口")
    parser.add_argument("--rate", type=float, default=20.0, help="轮询频率 Hz")
    parser.add_argument("--dry-run", action="store_true", help="不打开串口，只打印姿态")
    parser.add_argument("--yes", action="store_true", help="跳过启动确认")
    parser.add_argument("--force-start", action="store_true", help="允许从远离安全 stand 的姿态启动（危险）")
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate 必须大于 0")

    try:
        base = normalize_base(args.server, args.port)
    except ValueError as exc:
        print("[Error]", exc)
        return 1

    try:
        pose = wait_for_pose(base)
    except KeyboardInterrupt:
        print("\n[Info] 已取消")
        return 0
    print(f"[Info] 服务器姿态: {format_pose(pose)}")

    deltas = large_deltas(pose)
    if deltas:
        detail = ", ".join(
            f"{joint}号 {delta:+d} ({abs(delta) * 0.24:.0f}°)"
            for joint, delta in deltas.items()
        )
        print(f"[Warning] 初始姿态与安全 stand 差异较大: {detail}")
        print("[Warning] 这里假设机械臂当前在 stand，直接应用可能撞到箱子或拉紧线缆。")

    if args.dry_run:
        print("[Info] dry-run，不打开串口")
        try:
            run_dry(base, args.rate, pose)
        except KeyboardInterrupt:
            print("\n[Info] 已停止")
        return 0

    if deltas and args.yes and not args.force_start:
        print("[Error] --yes 不会跳过远姿态保护。")
        print("[Error] 请先在网页点“初始姿态”或重启 server；确需强制启动时加 --force-start。")
        return 1

    if not args.yes:
        if deltas:
            try:
                answer = input("仍然应用? [y/N] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer != "y":
                print("[Info] 已取消")
                return 0
        else:
            print("[Warning] 机械臂会移动到上面的姿态，请确认周围安全。")
            try:
                answer = input("将这组角度应用到机械臂? [Y/n] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer == "n":
                print("[Info] 已取消")
                return 0

    try:
        return run_arm(args, base, pose)
    except KeyboardInterrupt:
        print("\n[Info] 已停止")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
