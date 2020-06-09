# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
"""
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

from meshrcnn.data.datasets.pix3d import load_pix3d_json

__all__ = ["register_pix3d"]


def get_pix3d_metadata():
    meta = [
        {"name": "bed", "color": [255, 255, 25], "id": 1},  # noqa
        {"name": "bookcase", "color": [230, 25, 75], "id": 2},  # noqa
        {"name": "chair", "color": [250, 190, 190], "id": 3},  # noqa
        {"name": "desk", "color": [60, 180, 75], "id": 4},  # noqa
        #{"name": "misc", "color": [230, 190, 255], "id": 5},  # noqa
        {"name": "sofa", "color": [0, 130, 200], "id": 5},  # noqa
        {"name": "table", "color": [245, 130, 48], "id": 6},  # noqa
        #{"name": "tool", "color": [70, 240, 240], "id": 8},  # noqa
        {"name": "wardrobe", "color": [210, 245, 60], "id": 7},  # noqa
    ]
    return meta

def _register_pix3d(dataset_name, json_file, image_root, root="datasets"):
    PIX3D_KEYPOINT_NAMES = (
        "right_bottom_front",
        "left_bottom_front", "right_top_front",
        "left_top_front", "right_bottom_back",
        "left_bottom_back", "right_top_back",
        "left_top_back" , "center"
    )

    # Pairs of keypoints that should be exchanged under horizontal flipping
    PIX3D_KEYPOINT_FLIP_MAP = (
        ("right_bottom_front", "left_bottom_front"),
        ("left_bottom_front", "right_bottom_front"),
        ("right_top_front", "left_top_front"),
        ("left_top_front", "right_top_front"),
        ("right_bottom_back", "left_bottom_back"),
        ("left_bottom_back", "right_bottom_back"),
        ("right_top_back", "left_top_back"),
        ("left_top_back", "right_top_back"),
        ("center", "center"),
    )

    DatasetCatalog.register(
        dataset_name, lambda: load_pix3d_json(os.path.join(root, json_file),
                                              os.path.join(root, image_root), 
                                              dataset_name)
    )
    things_ids = [k["id"] for k in get_pix3d_metadata()]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in get_pix3d_metadata()]
    thing_colors = [k["color"] for k in get_pix3d_metadata()]
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
        "keypoint_names": PIX3D_KEYPOINT_NAMES,
        "keypoint_flip_map": PIX3D_KEYPOINT_FLIP_MAP,        
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="pix3d", **metadata
    )
def register_pix3d(root="datasets"):
    SPLITS = {
        "pix3d_s1_train": ("pix3d", "pix3d/pix3d_s1_train.json"),
        "pix3d_s1_test": ("pix3d", "pix3d/pix3d_s1_test.json"),
        "pix3d_s1_train_bbox8": ("pix3d", "pix3d/pix3d_s1_train_pruned_bbox8.json"),
        "pix3d_s1_test_bbox8": ("pix3d", "pix3d/pix3d_s1_test_pruned_bbox8.json"),

    }

    for key, (data_root, anno_file) in SPLITS.items():
        _register_pix3d(key, anno_file, data_root, root)