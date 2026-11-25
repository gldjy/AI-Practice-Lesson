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

from arm_params import format_pose, normalize_pose

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


def run_arm(args, base):
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

        pose = fetch_pose(base)
        print(f"[Info] 应用初始姿态: {format_pose(pose)}")
        apply_pose(uart, pose)
        last = pose
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

    if args.dry_run:
        print("[Info] dry-run，不打开串口")
        try:
            run_dry(base, args.rate, pose)
        except KeyboardInterrupt:
            print("\n[Info] 已停止")
        return 0

    if not args.yes:
        print("[Warning] 机械臂会移动到上面的姿态，请确认周围安全。")
        try:
            answer = input("将这组角度应用到机械臂? [Y/n] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "n":
            print("[Info] 已取消")
            return 0

    try:
        return run_arm(args, base)
    except KeyboardInterrupt:
        print("\n[Info] 已停止")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
