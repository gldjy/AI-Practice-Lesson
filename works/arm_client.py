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


def heartbeat(uart, stopped):
    """Sends the system heartbeat until the uart stops."""
    try:
        while uart.running and not stopped.is_set():
            uart.transmitSysHeart()
            stopped.wait(0.2)
    except Exception as exc:
        uart.io_error = exc


def check_uart(uart, require_connection=True):
    """Raises when the serial transport or controller communication fails."""
    if uart.io_error is not None:
        raise RuntimeError(f"串口收发失败: {uart.io_error}") from uart.io_error
    if not uart.running:
        raise RuntimeError("串口已停止")
    uart.comStatusCheck()
    if require_connection and not uart.connect:
        raise ConnectionError("下位机通信超时，停止发送机械臂动作")


def apply_pose(uart, pose, previous=None):
    """Sends a changed pose using the same action format as vendor stand."""
    changed = [
        joint
        for joint in sorted(pose)
        if previous is None or pose[joint] != previous.get(joint)
    ]
    if changed:
        check_uart(uart)
        # Matches the complete single-frame action used by vendor stand.
        row = tuple(pose[joint] for joint in range(1, 7)) + (0,)
        uart.setArmActions((row,))
        print(
            f"[Apply] {format_pose(pose)}  changed={changed}  action=1 wait=0",
            flush=True,
        )
    return changed


def vendor_stand_pose(uart):
    """Returns the final angles of the vendor stand action group."""
    frames = [frame for frame in uart.robotArm.actionStand if len(frame) > 6]
    return {joint: frames[-1][joint - 1] for joint in range(1, 7)}


def initialize_arm(uart):
    """Waits for controller traffic before enabling and sending vendor stand."""
    print("[Info] 等待下位机连接...", flush=True)
    uart.transmitSysHeart()
    deadline = time.monotonic() + 5.0
    while not uart.connect:
        check_uart(uart, require_connection=False)
        if time.monotonic() >= deadline:
            raise TimeoutError("5 秒内未收到下位机有效回包，未发送使能或 stand")
        time.sleep(0.05)
    print(
        f"[Info] 下位机已回包，错误码 0x{uart.sensors.errorCode:04X}", flush=True
    )
    time.sleep(1.0)  # Starts the boot delay after communication is established.
    check_uart(uart)
    uart.setEnable(True)
    time.sleep(0.3)  # Matches arm_test.py.
    check_uart(uart)
    print("[Info] 执行厂家 stand 初始化...", flush=True)
    uart.robotArmMoveit(uart.robotArm.Movement.stand)
    time.sleep(3.0)
    check_uart(uart)
    stand_pose = vendor_stand_pose(uart)
    print(f"[Info] 厂家 stand 已下发: {format_pose(stand_pose)}", flush=True)
    return stand_pose


STARTUP_DELTA = 100


def large_deltas(pose):
    """Returns joints that are far from the safe stand pose."""
    return {
        joint: pose[joint] - DEFAULT_POSE[joint]
        for joint in sorted(pose)
        if abs(pose[joint] - DEFAULT_POSE[joint]) >= STARTUP_DELTA
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
    uart.trace = args.trace_serial
    try:
        uart.start()
    except Exception as exc:
        print(f"[Error] 无法打开串口 {args.serial}: {exc}")
        return 1

    stopped = threading.Event()
    heart_thread = threading.Thread(target=heartbeat, args=(uart, stopped), daemon=True)
    reset_on_exit = False
    try:
        heart_thread.start()
        stand_pose = initialize_arm(uart)

        print(f"[Info] 调整到 server 姿态: {format_pose(target_pose)}", flush=True)
        apply_pose(uart, target_pose, stand_pose)
        last = target_pose
        reset_on_exit = True
        lost = False
        delay = 1.0 / args.rate
        print(f"[Info] 等待网页姿态更新: {base}/api/pose", flush=True)

        while True:
            check_uart(uart)
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

            # A new action replaces the previous target. Sending one every
            # poll restarts small moves constantly, so wait for the local
            # action window and then send only the latest pose.
            if pose != last and not uart.getArmActionsBusy():
                apply_pose(uart, pose, last)
                last = pose
            time.sleep(delay)
    except Exception as exc:
        reset_on_exit = False
        print(f"[Error] 控制循环异常: {exc}")
        return 1
    finally:
        try:
            if reset_on_exit and uart.io_error is None and uart.connect:
                print("[Info] 复位机械臂并关闭串口...")
                uart.stop(0, 0)
            else:
                print("[Info] 初始化中断或通信异常，关闭串口。")
        except Exception as exc:
            print(f"[Warning] 关闭串口时出错: {exc}")
        finally:
            stopped.set()
            uart.close()
            if heart_thread.ident is not None:
                heart_thread.join(timeout=1.5)


def main():
    parser = argparse.ArgumentParser(description="把 arm_server.py 的姿态同步到实体机械臂")
    parser.add_argument("--server", default="127.0.0.1", help="服务器主机或完整地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器未指定端口时使用")
    parser.add_argument("--serial", default="/dev/ttyUSB0", help="机械臂串口")
    parser.add_argument("--rate", type=float, default=20.0, help="轮询频率 Hz")
    parser.add_argument("--dry-run", action="store_true", help="不打开串口，只打印姿态")
    parser.add_argument("--yes", action="store_true", help="跳过启动确认")
    parser.add_argument("--trace-serial", action="store_true", help="打印实际发送的串口帧")
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

    if deltas and not args.force_start:
        print("[Error] 远姿态启动需要 --force-start，--yes 不会跳过保护。")
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
