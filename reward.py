import cv2
import numpy as np
from pathlib import Path

def crop(frame_bgr: np.ndarray, roi):
    x, y, w, h = roi
    return frame_bgr[y:y+h, x:x+w]

def bar_fill_ratio(bar_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((thr > 0).mean())

class TemplateMatcher:
    def __init__(self, template_path: str, threshold: float = 0.75):
        p = Path(template_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing template: {template_path}")
        self.template = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise RuntimeError(f"Failed to load template: {template_path}")
        self.thresh = threshold

    def score(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def matches(self, frame_bgr: np.ndarray) -> bool:
        return self.score(frame_bgr) >= self.thresh
