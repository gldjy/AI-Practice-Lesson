#!/usr/bin/env python
"""
机械臂串口接口的交互式验证脚本。

在实验箱上从 works/ 目录运行:
    python arm_test.py --port /dev/ttyUSB0

验证内容:
- 1~6 号舵机分别对应哪个部位，夹爪是几号
- 单舵机指令下发后动作是否平滑
- 手动点动时的步长、速度和安全范围
- 使能/失能指令是否有效
- 下位机是否回传机械臂角度
"""

import argparse
import re
import threading
import time

from uart import *

# Measured safe envelope for the current kit; see ArmParam.md.
JOINT_LIMITS = {
    1: (500, 665),  # Gripper: 640 grabs a cube.
    2: (145, 500),  # Upper bound not measured yet.
    3: (120, 500),
    4: (500, 860),
    5: (150, 740),
    6: (140, 860),
}

ROLE_NAMES = {
    "1": "底座旋转",
    "2": "大臂/肩部",
    "3": "小臂/肘部",
    "4": "腕部俯仰",
    "5": "腕部旋转",
    "6": "夹爪",
    "0": "没有可见动作",
    "?": "不确定",
}

# Addresses the arm commands use. They are transmit-only as far as this
# protocol implementation knows, so receiving one would mean angle feedback.
ARM_ADDRS = {0x1A: "动作组", 0x1B: "单舵机/使能"}


class ArmTester:
    """Runs the interactive checks against one Uart instance."""

    def __init__(self, uart, args):
        self.uart = uart
        self.step = args.step
        self.settle = args.settle
        self.limits = {} if args.no_limits else dict(JOINT_LIMITS)
        self.pose = None
        self.touched = {}
        self.addr_count = {}
        self.addr_lock = threading.Lock()
        self.enabled = None
        self._install_frame_spy()

    def _install_frame_spy(self):
        """Counts received frame addresses to check for arm feedback."""
        original = self.uart.transDataFrame

        def spy():
            if self.uart.recFinish:
                raw = self.uart.recBuff[1]
                addr = raw[0] if isinstance(raw, (bytes, bytearray)) else int(raw)
                with self.addr_lock:
                    self.addr_count[addr] = self.addr_count.get(addr, 0) + 1
            original()

        self.uart.transDataFrame = spy

    def clamp(self, joint, angle):
        """Clamps one joint angle to the protocol and test ranges."""
        angle = max(0, min(1000, int(angle)))
        if joint in self.limits:
            low, high = self.limits[joint]
            angle = max(low, min(high, angle))
        return angle

    def send_joint(self, joint, angle):
        """Sends one joint angle and updates the tracked pose."""
        target = self.clamp(joint, angle)
        if target != angle:
            print(f"[限位] {joint} 号舵机限制到 {target}")
        self.uart.setArmAngle(joint, target)
        self.pose[joint - 1] = target
        low, high = self.touched.get(joint, (target, target))
        self.touched[joint] = (min(low, target), max(high, target))
        return target

    def wait_idle(self, timeout=15.0):
        """Waits until the last action group should have finished."""
        deadline = time.time() + timeout
        while self.uart.getArmActionsBusy() and time.time() < deadline:
            time.sleep(0.05)

    def run_preset(self, name):
        """Runs a preset action group and syncs the tracked pose."""
        arm = self.uart.robotArm
        presets = {
            "stand": (arm.Movement.stand, arm.actionStand),
            "photo": (arm.Movement.photoTrans, arm.actionPhotoTrans),
            "reset": (arm.Movement.reset, arm.actionReset),
        }
        if name not in presets:
            print("可选动作组:", ", ".join(presets))
            return
        move, actions = presets[name]
        print(f"执行动作组 {name} ...")
        self.uart.robotArmMoveit(move)
        self.wait_idle()
        time.sleep(self.settle)
        # Skip the short padding rows some presets carry, such as actionStand.
        frames = [frame for frame in actions if len(frame) > 6]
        self.pose = list(frames[-1][:6])
        print("当前姿态记录为:", self.pose)

    def init_arm(self):
        """Enables the arm and moves it to the upright pose."""
        print("下一步会使能机械臂并执行 stand 动作组。")
        print("请确认机械臂周围和实验箱内没有障碍物。")
        input("按 Enter 继续，Ctrl+C 退出...")

        self.uart.transmitSysHeart()
        time.sleep(1.0)  # Wait for the lower controller to finish booting.
        self.uart.setEnable(True)
        self.enabled = True
        time.sleep(0.3)
        self.run_preset("stand")

        answer = input("机械臂是否已经变成直立姿态? [Y/n] ").strip().lower()
        if answer == "n":
            print("动作组没有生效，请检查串口、电源和使能指令。")
            return False
        return True

    def wiggle(self, joint):
        """Moves one joint by one step and returns it to the original angle."""
        origin = self.pose[joint - 1]
        low, high = self.limits.get(joint, (0, 1000))
        if not low <= origin <= high:
            print(f"{joint} 号舵机记录角度 {origin} 不在限位 {low}~{high} 内，跳过")
            return None
        if origin + self.step <= high:
            target = origin + self.step
        elif origin - self.step >= low:
            target = origin - self.step
        else:
            print(f"{joint} 号舵机没有可测试的余量")
            return None

        print(f"{joint} 号舵机: {origin} -> {target} -> {origin}")
        self.send_joint(joint, target)
        time.sleep(1.0)
        self.send_joint(joint, origin)
        time.sleep(1.0)
        return origin

    def identify(self):
        """Wiggles each joint so the user can report which part it drives."""
        print(f"每个舵机将移动 {self.step} 个单位，约 {self.step * 0.24:.1f}°，然后回到原位。")
        roles = {}
        for joint in range(1, 7):
            input(f"准备测试 {joint} 号舵机，确认周围安全后按 Enter...")
            if self.wiggle(joint) is None:
                roles[joint] = "无法测试"
                continue
            answer = input(
                "哪个部位动了? "
                "[1]底座旋转 [2]大臂 [3]小臂 [4]腕部俯仰 "
                "[5]腕部旋转 [6]夹爪 [0]没动 [?]不确定: "
            ).strip()
            roles[joint] = ROLE_NAMES.get(answer, answer or "?")

        print("\n舵机编号与部位:")
        for joint, role in roles.items():
            print(f"  {joint} 号: {role}")

        smooth = input("动作是连续平滑还是明显分段? [1]平滑 [2]跳变 [3]没看清: ").strip()
        print("平滑度:", {"1": "平滑", "2": "分段/跳变"}.get(smooth, "未判断"))

    def jog(self):
        """Jogs one joint under manual control and records angles sent."""
        raw = input("选择关节 1~6: ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= 6:
            print("输入无效")
            return
        joint = int(raw)
        low, high = self.limits.get(joint, (0, 1000))
        step = self.step
        print("a 减小, d 增大, a5/d5 连续 5 步, +10/-10 调整步长, p 打印姿态, q 返回")
        while True:
            angle = self.pose[joint - 1]
            command = input(
                f"[{joint} 号] {angle} ({low}~{high}) step={step} > "
            ).strip().lower()
            if command == "q":
                return
            if command == "p":
                self.show_state()
                continue
            if re.fullmatch(r"[+-]\d+", command):
                step = max(1, step + int(command))
                print("步长改为", step)
                continue
            match = re.fullmatch(r"([ad])(\d*)", command)
            if not match:
                print("无法识别")
                continue

            direction = -1 if match.group(1) == "a" else 1
            for _ in range(int(match.group(2) or 1)):
                current = self.pose[joint - 1]
                target = self.clamp(joint, current + direction * step)
                if target == current:
                    print("已到限位", current)
                    break
                self.send_joint(joint, target)
                print(" ->", target)
                time.sleep(0.15)

    def toggle_enable(self):
        """Disables the arm briefly so the user can feel the torque drop."""
        print("失能后机械臂可能因重力下坠，请先扶住或确认姿态安全。")
        self.uart.setEnable(False)
        self.enabled = False
        print("已失能。试着用手扳动机械臂，应该能扳动。")
        input("观察后按 Enter 重新使能...")
        self.uart.setEnable(True)
        self.enabled = True
        print("已重新使能，试着扳动机械臂，应该扳不动。")

    def show_frames(self):
        """Prints the addresses received so far."""
        with self.addr_lock:
            items = sorted(self.addr_count.items())
        if not items:
            print("还没有收到任何数据帧")
            return
        print("已收到的地址:")
        for addr, count in items:
            name = ARM_ADDRS.get(addr, "")
            print(f"  0x{addr:02X} x{count} {name}")
        if not any(addr in ARM_ADDRS for addr in self.addr_count):
            print("没有出现 0x1A/0x1B，说明下位机没有回传机械臂角度。")

    def show_state(self):
        """Prints the tracked pose and the angles actually sent."""
        print("记录姿态:", self.pose)
        if self.touched:
            print("本次实际下发过的角度:")
            for joint in sorted(self.touched):
                low, high = self.touched[joint]
                print(f"  {joint} 号: {low} ~ {high}")

    def choose_preset(self):
        """Asks for a preset name and runs it."""
        name = input("动作组 stand/photo/reset: ").strip()
        self.run_preset(name)

    def run(self):
        """Runs the interactive command loop."""
        if not self.init_arm():
            return
        commands = {
            "m": self.identify,
            "j": self.jog,
            "p": self.choose_preset,
            "e": self.toggle_enable,
            "f": self.show_frames,
            "l": self.show_state,
        }
        while True:
            print("\n命令: m 识别舵机  j 点动  p 动作组  e 使能测试  f 通信地址  l 状态  q 退出")
            command = input("> ").strip().lower()
            if command == "q":
                return
            action = commands.get(command)
            if action:
                action()
            else:
                print("未知命令")


def heartbeat(uart):
    """Sends the system heartbeat until the uart stops."""
    while uart.running:
        uart.transmitSysHeart()
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description="机械臂串口接口的交互式验证脚本")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--step", type=int, default=20, help="单次点动的角度单位 0~1000")
    parser.add_argument("--settle", type=float, default=3.0, help="动作组结束后的等待时间: s")
    parser.add_argument("--no-limits", action="store_true", help="不使用预设角度范围做限制")
    parser.add_argument("--dry-run", action="store_true", help="不打开串口，只打印要发送的帧")
    args = parser.parse_args()

    uart = Uart(args.port)
    if args.dry_run:
        uart.running = True

        def fake_transmit(addr, data):
            print(
                f"[dry-run] addr=0x{int.from_bytes(addr, 'big'):02X} "
                f"data={'None' if data is None else data.hex()}"
            )

        uart.transmitFrame = fake_transmit
    else:
        try:
            uart.start()
        except Exception as exc:
            print("无法打开串口:", exc)
            return 1
        threading.Thread(target=heartbeat, args=(uart,), daemon=True).start()

    tester = ArmTester(uart, args)
    try:
        tester.run()
    finally:
        if not args.dry_run:
            print("退出前复位机械臂并关闭串口...")
            try:
                uart.stop(0, 0)
            except Exception as exc:
                print("关闭串口时出错:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
