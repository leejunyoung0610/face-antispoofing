from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


def _expand_bbox(bbox, img_size, margin=1.3):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    new_w = w * margin
    new_h = h * margin
    x1 = max(int(cx - new_w / 2), 0)
    y1 = max(int(cy - new_h / 2), 0)
    x2 = min(int(cx + new_w / 2), img_size[0])
    y2 = min(int(cy + new_h / 2), img_size[1])
    return x1, y1, x2, y2


@lru_cache(maxsize=1)
def _load_haar():
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return haar


def _detect_with_mediapipe(image):
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    img_rgb = image.convert("RGB")
    results = mp_face.process(cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR))
    mp_face.close()
    if not results.detections:
        return None
    detection = results.detections[0].location_data.relative_bounding_box
    w, h = image.size
    x = detection.xmin * w
    y = detection.ymin * h
    width = detection.width * w
    height = detection.height * h
    return int(x), int(y), int(width), int(height)


def _detect_with_haar(image):
    haar = _load_haar()
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return None
    return tuple(faces[0])


def crop_face_pil(image, size=(224, 224), margin=1.3, return_status=False):
    try:
        bbox = None
        if MP_AVAILABLE:
            bbox = _detect_with_mediapipe(image)
        if bbox is None:
            bbox = _detect_with_haar(image)
        if bbox is None:
            resized = image.resize(size, Image.BILINEAR)
            return resized if not return_status else (resized, False)
        x1, y1, x2, y2 = _expand_bbox(bbox, image.size, margin)
        crop = image.crop((x1, y1, x2, y2))
        resized = crop.resize(size, Image.BILINEAR)
        return resized if not return_status else (resized, True)
    except Exception:
        resized = image.resize(size, Image.BILINEAR)
        return resized if not return_status else (resized, False)
