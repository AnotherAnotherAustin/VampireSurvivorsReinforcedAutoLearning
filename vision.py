import cv2
import numpy as np
from pathlib import Path

class PlayerTracker:
    def __init__(self, template_path: str, threshold: float = 0.72, search_radius: int = 220):
        p = Path(template_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing player template: {template_path}")
        tpl = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            raise RuntimeError(f"Failed to load player template: {template_path}")
        self.tpl = tpl
        self.thresh = float(threshold)
        self.search_radius = int(search_radius)
        self.last_xy = None
        self.th, self.tw = self.tpl.shape[:2]

    def _roi_around_last(self, frame_shape):
        H, W = frame_shape[:2]
        cx, cy = self.last_xy
        r = self.search_radius
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(W, cx + r)
        y2 = min(H, cy + r)
        return x1, y1, x2, y2

    def locate(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.last_xy is not None:
            x1, y1, x2, y2 = self._roi_around_last(gray.shape)
            search = gray[y1:y2, x1:x2]
            res = cv2.matchTemplate(search, self.tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.thresh:
                px = x1 + max_loc[0]
                py = y1 + max_loc[1]
                cx = px + self.tw // 2
                cy = py + self.th // 2
                self.last_xy = (cx, cy)
                return cx, cy, float(max_val)

        res = cv2.matchTemplate(gray, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= self.thresh:
            px, py = max_loc
            cx = px + self.tw // 2
            cy = py + self.th // 2
            self.last_xy = (cx, cy)
            return cx, cy, float(max_val)

        return None, None, float(max_val)

def enemy_density_ring(frame_bgr: np.ndarray, cx: int, cy: int, r_in: int, r_out: int) -> float:
    H, W = frame_bgr.shape[:2]
    x1 = max(0, cx - r_out)
    y1 = max(0, cy - r_out)
    x2 = min(W, cx + r_out)
    y2 = min(H, cy + r_out)
    patch = frame_bgr[y1:y2, x1:x2]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    diff = cv2.subtract(blur, gray)
    _, mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)

    ph, pw = gray.shape
    yy, xx = np.ogrid[:ph, :pw]
    ccx = cx - x1
    ccy = cy - y1
    dist2 = (xx - ccx) ** 2 + (yy - ccy) ** 2
    ring = (dist2 >= r_in * r_in) & (dist2 <= r_out * r_out)

    ring_mask = mask[ring]
    if ring_mask.size == 0:
        return 0.0
    return float((ring_mask > 0).mean())
