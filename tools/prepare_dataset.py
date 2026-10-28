#!/usr/bin/env python3
# prepare_dataset.py --zip <labeled.zip> [--out data/<dataset>]
# Converts a labeled export (images + one XML per image) into a COCO dataset
# that paddet can read:
#   <out>/images/*.jpg
#   <out>/annotations/instance_train.json
#   <out>/annotations/instance_val.json

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

VAL_RATIO = 0.1
SEED = 0


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="labeled export archive")
    parser.add_argument("--out", default="data/dataset", help="output dataset dir")
    args = parser.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    annotations_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

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
            member = next(n for n in names if Path(n).name == image_name and "/Images/" in n)
            target = images_dir / image_name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            samples.append({"name": image_name, **parsed})

    samples.sort(key=lambda s: s["name"])
    categories = sorted({box[0] for sample in samples for box in sample["boxes"]})

    shuffled = samples.copy()
    random.Random(SEED).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * VAL_RATIO))
    val = shuffled[:val_count]
    train = shuffled[val_count:]

    for split, split_samples in (("train", train), ("val", val)):
        path = annotations_dir / f"instance_{split}.json"
        path.write_text(json.dumps(build_coco(split_samples, categories), indent=1))

    print(f"images  : {len(samples)}")
    print(f"classes : {categories}")
    print(f"train   : {len(train)} -> {annotations_dir / 'instance_train.json'}")
    print(f"val     : {len(val)} -> {annotations_dir / 'instance_val.json'}")


if __name__ == "__main__":
    main()
