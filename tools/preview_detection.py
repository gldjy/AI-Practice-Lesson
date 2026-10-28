#!/usr/bin/env python3
# preview_detection.py --model-dir <exported dir> [--source 0] [--save out.mp4]
# Runs an exported PaddleDetection model through Paddle Inference and shows the
# detections live. Works with a camera, a video file or a single image, and
# needs nothing from paddet, so it also runs on the deploy group.

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import paddle.inference as paddle_infer
import yaml

PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
]


def resolve_model_dir(path: str) -> Path:
    """Returns the directory that actually contains model.pdmodel."""
    model_dir = Path(path)
    if (model_dir / "model.pdmodel").is_file():
        return model_dir
    candidates = [p.parent for p in model_dir.glob("*/model.pdmodel")]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(f"no model.pdmodel in {model_dir}, export one with tools/export_model.py")


def load_infer_config(model_dir: Path) -> dict:
    with (model_dir / "infer_cfg.yml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def create_predictor(model_dir: Path, device: str, threads: int):
    """Builds a Paddle Inference predictor for the exported model."""
    config = paddle_infer.Config(
        str(model_dir / "model.pdmodel"), str(model_dir / "model.pdiparams")
    )
    config.disable_glog_info()
    config.switch_ir_optim(True)
    if device.upper() == "GPU":
        config.enable_use_gpu(500, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(threads)
    return paddle_infer.create_predictor(config)


def preprocess(frame: np.ndarray, infer_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Applies the Resize/Normalize/Permute chain recorded in infer_cfg.yml."""
    # paddet trains on RGB, OpenCV hands out BGR.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for op in infer_cfg["Preprocess"]:
        kind = op["type"]
        if kind == "Resize":
            height, width = op["target_size"]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        elif kind == "NormalizeImage":
            image = image.astype("float32")
            scale = 1.0 / 255.0 if op.get("is_scale", True) else 1.0
            mean = np.array(op["mean"], dtype="float32")
            std = np.array(op["std"], dtype="float32")
            image = (image * scale - mean) / std
        elif kind == "Permute":
            image = image.transpose(2, 0, 1)
        else:
            raise SystemExit(f"unsupported preprocess op: {kind}")

    blob = np.ascontiguousarray(image[None], dtype="float32")
    target_h, target_w = infer_cfg["Preprocess"][0]["target_size"]
    scale_factor = np.array(
        [[frame.shape[0] / target_h, frame.shape[1] / target_w]], dtype="float32"
    )
    return blob, scale_factor


def detect(predictor, frame: np.ndarray, infer_cfg: dict, threshold: float) -> list[dict]:
    """Runs one frame and returns the boxes above the score threshold.

    Static-shape exports bake the input size into the graph, so their NMS works
    in the resized coordinate system and the boxes have to be mapped back here.
    Dynamic-shape exports take `im_shape` and already return original-size boxes.
    """
    blob, scale_factor = preprocess(frame, infer_cfg)
    input_names = predictor.get_input_names()
    dynamic_shape = "im_shape" in input_names
    images_info = np.array([[frame.shape[0], frame.shape[1]]], dtype="float32")
    payloads = {
        "image": blob,
        # A graph that ignores this feed accepts any value, ones keeps the
        # output in resized space for the manual mapping below.
        "scale_factor": scale_factor if dynamic_shape else np.ones_like(scale_factor),
        "im_shape": images_info,
    }
    for name in input_names:
        predictor.get_input_handle(name).copy_from_cpu(payloads[name])
    predictor.run()

    outputs = predictor.get_output_names()
    boxes = predictor.get_output_handle(outputs[0]).copy_to_cpu()
    count = int(predictor.get_output_handle(outputs[1]).copy_to_cpu()[0]) if len(outputs) > 1 else len(boxes)

    target_h, target_w = infer_cfg["Preprocess"][0]["target_size"]
    scale_x = 1.0 if dynamic_shape else frame.shape[1] / target_w
    scale_y = 1.0 if dynamic_shape else frame.shape[0] / target_h

    labels = infer_cfg["label_list"]
    detections = []
    for class_id, score, x1, y1, x2, y2 in boxes[:count]:
        if score < threshold:
            continue
        detections.append({
            "label": labels[int(class_id)],
            "score": float(score),
            "box": (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y),
            ),
        })
    return detections


def draw(frame: np.ndarray, detections: list[dict], fps: float) -> np.ndarray:
    """Draws boxes, labels and the frame rate on a copy of the frame."""
    canvas = frame.copy()
    for index, detection in enumerate(detections):
        color = PALETTE[index % len(PALETTE)]
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = f"{detection['label']} {detection['score']:.2f}"
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # prefer above the box, drop inside when the box touches the top edge
        top = y1 - height - baseline - 6
        if top < 0:
            top = y1 + 2
        cv2.rectangle(canvas, (x1, top), (x1 + width + 6, top + height + baseline + 4),
                      color, -1)
        cv2.putText(canvas, text, (x1 + 3, top + height + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    height = canvas.shape[0]
    cv2.putText(canvas, f"{fps:5.1f} fps  {len(detections)} obj", (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def open_source(source: str):
    """Returns a capture for a camera index or video, or None for a still image."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        if not capture.isOpened():
            raise SystemExit(f"cannot open camera {source}")
        return capture
    if not Path(source).is_file():
        raise SystemExit(f"{source} does not exist")
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return None
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video {source}")
    return capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="directory with model.pdmodel, model.pdiparams, infer_cfg.yml")
    parser.add_argument("--source", default="0",
                        help="camera index, video path or image path (default: 0)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--threads", type=int, default=4, help="CPU inference threads")
    parser.add_argument("--threshold", type=float, default=None,
                        help="score threshold, defaults to draw_threshold from infer_cfg.yml")
    parser.add_argument("--save", default=None, help="write the result to this image or video")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    infer_cfg = load_infer_config(model_dir)
    threshold = args.threshold if args.threshold is not None else float(infer_cfg.get("draw_threshold", 0.5))
    predictor = create_predictor(model_dir, args.device, args.threads)
    print(f"model     : {model_dir}")
    print(f"classes   : {infer_cfg['label_list']}")
    print(f"device    : {args.device}, threshold {threshold}")

    capture = open_source(args.source)
    writer = None
    frame_times: list[float] = []
    processed = 0

    try:
        while True:
            if capture is None:
                frame = cv2.imread(args.source)
                if frame is None:
                    raise SystemExit(f"cannot read image {args.source}")
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            start = time.perf_counter()
            detections = detect(predictor, frame, infer_cfg, threshold)
            frame_times = (frame_times + [time.perf_counter() - start])[-30:]
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            canvas = draw(frame, detections, fps)

            if args.save:
                if capture is None:
                    cv2.imwrite(args.save, canvas)
                else:
                    if writer is None:
                        height, width = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                            capture.get(cv2.CAP_PROP_FPS) or 25.0, (width, height))
                    writer.write(canvas)
            if not args.no_show:
                cv2.imshow("preview_detection", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            processed += 1
            if capture is None or (args.max_frames and processed >= args.max_frames):
                break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"processed : {processed} frame(s), last fps {fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
