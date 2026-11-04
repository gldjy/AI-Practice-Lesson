#!/usr/bin/env python3
# prepare_dataset.py --zip <labeled.zip> [--out data/<dataset>] [--augment]
# Converts a labeled export (images + one XML per image) into a COCO dataset
# that paddet can read:
#   <out>/images/*.jpg
#   <out>/annotations/instance_train.json
#   <out>/annotations/instance_val.json
# With --augment, every training image also gets three variants:
#   <stem>_aux_1.jpg  horizontal flip
#   <stem>_aux_2.jpg  vertical flip
#   <stem>_aux_3.jpg  white balance shift

import argparse
import hashlib
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

VAL_RATIO = 0.1
SEED = 0
WHITE_BALANCE_STRENGTH = 0.15
AUX_PATTERN = re.compile(r"_aux_\d+$")


def parse_annotation(xml_bytes: bytes) -> dict:
    """Parses one annotation file into image size and object boxes."""
    root = ElementTree.fromstring(xml_bytes)
    size = root.find("size")
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")
        boxes.append((
            name,
            int(box.findtext("xmin")),
            int(box.findtext("ymin")),
            int(box.findtext("xmax")),
            int(box.findtext("ymax")),
        ))
    return {
        "width": int(size.findtext("width")),
        "height": int(size.findtext("height")),
        "boxes": boxes,
    }


def build_coco(samples: list[dict], categories: list[str]) -> dict:
    """Builds the COCO dictionary for a list of parsed samples."""
    cat_ids = {name: index + 1 for index, name in enumerate(categories)}
    images, annotations = [], []
    for image_id, sample in enumerate(samples, start=1):
        name = sample["name"]
        images.append({
            "id": image_id,
            "file_name": name,
            "width": sample["width"],
            "height": sample["height"],
        })
        for category, xmin, ymin, xmax, ymax in sample["boxes"]:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": cat_ids[category],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
            })
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cat_id, "name": name, "supercategory": "cube"}
            for name, cat_id in cat_ids.items()
        ],
    }


def family_of(name: str) -> str:
    """Returns the original image stem without an augmentation suffix."""
    return AUX_PATTERN.sub("", Path(name).stem)


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits samples by original image family, keeping augmented copies together."""
    shuffled = samples.copy()
    random.Random(SEED).shuffle(shuffled)

    families = []
    seen = set()
    for sample in shuffled:
        family = family_of(sample["name"])
        if family not in seen:
            seen.add(family)
            families.append(family)

    val_count = max(1, round(len(families) * VAL_RATIO))
    val_families = set(families[:val_count])
    train = [s for s in shuffled if family_of(s["name"]) not in val_families]
    val = [s for s in shuffled if family_of(s["name"]) in val_families]
    return train, val


def clamp_boxes(boxes: list[tuple], width: int, height: int) -> list[tuple]:
    """Keeps boxes inside the image and ensures a positive width and height."""
    clamped = []
    for name, x1, y1, x2, y2 in boxes:
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(x1 + 1, min(width, int(x2)))
        y2 = max(y1 + 1, min(height, int(y2)))
        clamped.append((name, x1, y1, x2, y2))
    return clamped


def horizontal_flip(image: np.ndarray, boxes: list[tuple]) -> tuple[np.ndarray, list[tuple]]:
    """Flips an image left to right and mirrors the boxes."""
    height, width = image.shape[:2]
    image = cv2.flip(image, 1)
    boxes = [
        (name, width - x2, y1, width - x1, y2)
        for name, x1, y1, x2, y2 in boxes
    ]
    return image, clamp_boxes(boxes, width, height)


def vertical_flip(image: np.ndarray, boxes: list[tuple]) -> tuple[np.ndarray, list[tuple]]:
    """Flips an image upside down and mirrors the boxes."""
    height, width = image.shape[:2]
    image = cv2.flip(image, 0)
    boxes = [
        (name, x1, height - y2, x2, height - y1)
        for name, x1, y1, x2, y2 in boxes
    ]
    return image, clamp_boxes(boxes, width, height)


def white_balance(image: np.ndarray, name: str) -> np.ndarray:
    """Shifts the red and blue balance deterministically from the image name."""
    digest = hashlib.sha256(f"{SEED}:{family_of(name)}:white_balance".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    shift = value * 2.0 - 1.0
    red_gain = 1.0 + WHITE_BALANCE_STRENGTH * shift
    blue_gain = 1.0 - WHITE_BALANCE_STRENGTH * shift

    result = image.astype(np.float32)
    result[..., 2] *= red_gain
    result[..., 0] *= blue_gain
    return np.clip(result, 0, 255).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    """Reads one image in BGR order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Writes one image, failing loudly when the encoder rejects it."""
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"cannot write image: {path}")


def augment_samples(samples: list[dict], images_dir: Path) -> list[dict]:
    """Writes three augmented copies for every sample and returns their records."""
    augmented = []
    for sample in samples:
        image = read_image(images_dir / sample["name"])
        height, width = image.shape[:2]
        if (width, height) != (sample["width"], sample["height"]):
            raise ValueError(
                f"annotation size {sample['width']}x{sample['height']} does not match "
                f"image size {width}x{height}: {sample['name']}"
            )

        variants = [
            horizontal_flip(image, sample["boxes"]),
            vertical_flip(image, sample["boxes"]),
            (white_balance(image, sample["name"]), sample["boxes"]),
        ]

        stem = Path(sample["name"]).stem
        for index, (variant, boxes) in enumerate(variants, start=1):
            name = f"{stem}_aux_{index}.jpg"
            write_image(images_dir / name, variant)
            augmented.append({
                "name": name,
                "width": width,
                "height": height,
                "boxes": boxes,
            })
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="labeled export archive")
    parser.add_argument("--out", default="data/dataset", help="output dataset dir")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="write horizontal flip, vertical flip and white balance copies for each training image",
    )
    args = parser.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    annotations_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Remove variants from an earlier run so the output matches this invocation.
    for stale in images_dir.glob("*_aux_*.jpg"):
        stale.unlink()

    samples = []
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        annotations = sorted(n for n in names if "/Annotations/" in n and n.endswith(".xml"))

        for anno_name in annotations:
            parsed = parse_annotation(archive.read(anno_name))
            stem = Path(anno_name).stem
            image_name = next(
                (Path(n).name for n in names
                 if "/Images/" in n and Path(n).stem == stem and n.lower().endswith((".jpg", ".jpeg", ".png"))),
                None,
            )
            if image_name is None:
                raise FileNotFoundError(f"no image found for {anno_name}")
            if AUX_PATTERN.search(Path(image_name).stem):
                raise ValueError(f"input image already has an augmentation suffix: {image_name}")

            member = next(n for n in names if Path(n).name == image_name and "/Images/" in n)
            target = images_dir / image_name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            samples.append({"name": image_name, **parsed})

    samples.sort(key=lambda s: s["name"])
    categories = sorted({box[0] for sample in samples for box in sample["boxes"]})

    train, val = split_samples(samples)
    train_families = {family_of(sample["name"]) for sample in train}
    val_families = {family_of(sample["name"]) for sample in val}
    if train_families & val_families:
        raise RuntimeError("train and val contain the same image family")

    augmented = augment_samples(train, images_dir) if args.augment else []
    train_output = train + augmented

    if any(AUX_PATTERN.search(Path(sample["name"]).stem) for sample in val):
        raise RuntimeError("val contains an augmented image")

    for split, split_data in (("train", train_output), ("val", val)):
        path = annotations_dir / f"instance_{split}.json"
        path.write_text(json.dumps(build_coco(split_data, categories), indent=1))

    print(f"originals : {len(samples)}")
    print(f"augmented : {len(augmented)}")
    print(f"classes   : {categories}")
    print(f"train     : {len(train_output)} -> {annotations_dir / 'instance_train.json'}")
    print(f"val       : {len(val)} -> {annotations_dir / 'instance_val.json'}")


if __name__ == "__main__":
    main()
