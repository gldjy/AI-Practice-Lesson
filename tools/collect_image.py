#!/usr/bin/env python3
# collect_image.py <device> <save_dir>

import os
import re
import sys
from typing import Set

import cv2

WIDTH = 640
HEIGHT = 480

IMAGE_NAME_PATTERN = re.compile(r"^image_(\d+)\.jpg$")


def scan_used_indices(save_dir: str) -> Set[int]:
    if not os.path.isdir(save_dir):
        return set()

    used = set()
    for name in os.listdir(save_dir):
        match = IMAGE_NAME_PATTERN.match(name)
        if match:
            used.add(int(match.group(1)))
    return used


def next_free_index(used: Set[int], start: int = 0) -> int:
    index = start
    while index in used:
        index += 1
    return index


def video(capture: cv2.VideoCapture, save_dir: str):
    used = scan_used_indices(save_dir)
    index = next_free_index(used)
    print(f"Found {len(used)} existing images, next index is {index:04d}")

    codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capture.set(cv2.CAP_PROP_FOURCC, codec)
    capture.set(cv2.CAP_PROP_FPS, 60)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    WINDOW_NAME = "Frame"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    window_visible = False

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            window_visible = True
        elif window_visible:
            break

        frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Space
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, f"image_{index:04d}.jpg")
            cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved image to {file_path}")

            used.add(index)
            index = next_free_index(used, index + 1)
        elif key == 27:  # Esc
            break

        cv2.putText(
            frame,
            f"#{index:04d}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "You can press [Space] to collect images",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {os.path.basename(__file__)} <device> <save_dir>")
        sys.exit(1)

    dev, save_dir = sys.argv[1], sys.argv[2]
    print("Starting")
    capture = cv2.VideoCapture(dev)
    video(capture, save_dir)
