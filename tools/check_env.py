#!/usr/bin/env python3
# check_env.py [-c config] [--cpu]
# Reports the CUDA/GPU capability of this machine and verifies that both
# paddle and the paddet (PaddleDetection) submodule actually work.

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PADDET_ROOT = ROOT / "vendors" / "paddet"
DEFAULT_CONFIG = "configs/yolov3/yolov3_darknet53_270e_coco.yml"

failures: list[str] = []


def section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def ok(message: str) -> None:
    print(f"  [ok]   {message}")


def fail(message: str, reason: str) -> None:
    print(f"  [fail] {message}: {reason}")
    failures.append(message)


def run(cmd: list[str], timeout: int = 20) -> str | None:
    """Runs a command and returns its stdout, or None when it is unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def read_file(path: str) -> str:
    """Returns a small /proc or /sys file as text, or 'unknown' when unreadable."""
    try:
        return Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return "unknown"


def total_memory_gib() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def report_system() -> None:
    section("system")
    print(f"  python     : {sys.version.split()[0]} ({sys.executable})")

    if Path("/etc/nv_tegra_release").is_file():
        model = read_file("/proc/device-tree/model")
        release = read_file("/etc/nv_tegra_release")
        print(f"  board      : {model}")
        print(f"  jetpack    : {release}")
        print(f"  vram       : {total_memory_gib():.1f} GiB unified with the cpu")
        print("  nvidia-smi : Tegra has none, tegrastats is the equivalent")
        return

    nvidia_smi = run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if nvidia_smi is None:
        print("  nvidia-smi : not available, assuming a CPU-only machine")
        return

    for line in nvidia_smi.splitlines():
        index, name, driver, memory = [part.strip() for part in line.split(",")]
        print(f"  gpu {index:>2}     : {name} | {int(memory) / 1024:.1f} GiB | driver {driver}")

    header = run(["nvidia-smi"]) or ""
    # Newer drivers print "CUDA UMD Version", older ones "CUDA Version".
    match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", header)
    if match:
        print(f"  driver cuda: {match.group(1)} (highest runtime the driver accepts)")

    nvcc = run(["nvcc", "--version"])
    match = re.search(r"release ([\d.]+)", nvcc or "")
    if match:
        print(f"  toolkit    : CUDA {match.group(1)} (nvcc on PATH)")
    else:
        print("  toolkit    : no nvcc on PATH (not needed, the wheel bundles its own runtime)")


def report_paddle(paddle, expect_gpu: bool = True) -> bool:
    section("paddle")
    ok(f"version {paddle.__version__}")

    try:
        cuda_version = paddle.version.cuda()
        cudnn_version = paddle.version.cudnn()
    except Exception as exc:  # noqa: BLE001 - paddle raises on CPU builds
        cuda_version, cudnn_version = "n/a", "n/a"
        print(f"  note       : cannot read runtime versions ({exc})")
    print(f"  runtime    : cuda {cuda_version} | cudnn {cudnn_version}")

    if not paddle.device.is_compiled_with_cuda():
        if expect_gpu:
            fail(
                "cuda build",
                "paddle is compiled without CUDA, install paddlepaddle-gpu "
                "(impossible on arm64, pass --cpu there)",
            )
        else:
            ok("cpu build as requested")
        return False

    count = paddle.device.cuda.device_count()
    if count == 0:
        fail("gpu visible", "no device reported by paddle")
        return False

    total_memory = 0
    for index in range(count):
        props = paddle.device.cuda.get_device_properties(index)
        capability = paddle.device.cuda.get_device_capability(index)
        total_memory += props.total_memory
        print(
            f"  gpu {index:>2}     : {props.name} | "
            f"{props.total_memory / 1024**3:.1f} GiB | "
            f"sm_{capability[0]}{capability[1]} | "
            f"{props.multi_processor_count} SMs"
        )
    print(f"  total vram : {total_memory / 1024**3:.1f} GiB across {count} device(s)")

    try:
        paddle.utils.run_check()
        ok("paddle self-check passed")
    except Exception as exc:  # noqa: BLE001 - run_check raises on a broken install
        lines = [line for line in str(exc).splitlines() if line.strip()]
        fail("paddle self-check", lines[-1] if lines else repr(exc))
        if sys.platform == "win32" and "cudnn" in str(exc).lower():
            print("  hint       : Windows wheels do not bundle CUDA, install CUDA 11.8")
            print("               and cuDNN 8.6 for CUDA 11.x, then add both bin dirs to PATH")
        return False

    return True


def report_ppdet(paddle, config: str, use_gpu: bool) -> None:
    section("paddet")
    try:
        import ppdet
        from ppdet.core.workspace import create, load_config
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "ppdet":
            print("  note       : paddet is not installed, this is the deploy group")
            print("               use tools/preview_detection.py to run an exported model")
            return
        fail("import ppdet", repr(exc))
        print("  hint       : uv sync --group train")
        return
    version = getattr(ppdet, "__version__", "unknown")
    commit = getattr(ppdet, "__git_commit__", "unknown")
    ok(f"import ppdet {version} @ {commit[:8]}")

    config_path = PADDET_ROOT / config
    if not config_path.is_file():
        fail("load config", f"{config_path} does not exist")
        return

    paddle.set_device("gpu:0" if use_gpu else "cpu")
    try:
        cfg = load_config(str(config_path))
        model = create(cfg["architecture"])
        model.eval()
    except Exception as exc:  # noqa: BLE001 - building the graph is the actual test
        fail(f"build model from {config}", repr(exc))
        return
    ok(f"build model from {config}")

    image = paddle.randn([1, 3, 320, 320])
    inputs = {
        "image": image,
        "im_shape": paddle.to_tensor([[320.0, 320.0]]),
        "scale_factor": paddle.to_tensor([[1.0, 1.0]]),
    }
    try:
        with paddle.no_grad():
            model(inputs)  # warm up lazy kernels and cudnn autotuning
            start = time.perf_counter()
            outputs = model(inputs)
            if use_gpu:
                paddle.device.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
    except Exception as exc:  # noqa: BLE001 - forward pass exercises real kernels
        fail("forward pass", repr(exc))
        return

    shape = tuple(outputs["bbox"].shape) if isinstance(outputs, dict) and "bbox" in outputs else None
    detail = f", bbox {shape}" if shape else ""
    ok(f"forward pass on {paddle.get_device()} in {elapsed:.0f} ms{detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help=f"config relative to vendors/paddet (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="run the model test on CPU even when a GPU is available",
    )
    args = parser.parse_args()

    report_system()

    try:
        import paddle
    except Exception as exc:  # noqa: BLE001 - the venv is simply not ready yet
        fail("import paddle", repr(exc))
        print("  hint       : uv sync")
        paddle = None

    if paddle is not None:
        use_gpu = report_paddle(paddle, expect_gpu=not args.cpu)
        report_ppdet(paddle, args.config, use_gpu and not args.cpu)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
