# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, obb, pose, segment, detect_nmsfree

from .model import YOLO

__all__ = "classify", "segment", "detect", "pose", "obb", "detect_nmsfree", "YOLO"
