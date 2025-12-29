import numpy as np
from mss import mss

# OpenCV is optional but strongly recommended for fast resize / grayscale.
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None
    _cv2_err = e


class MonitorCapture:
    """Captures full-monitor frames using MSS.

    Returns BGR uint8 frames shaped (H, W, 3).
    """

    def __init__(self, monitor_index: int = 1):
        self.sct = mss()
        # Clamp monitor index to a valid range so we don't crash on startup.
        max_idx = len(self.sct.monitors) - 1  # monitors[0] is 'all'
        if monitor_index < 1 or monitor_index > max_idx:
            monitor_index = 1
        self.monitor_index = monitor_index

    def grab(self) -> np.ndarray:
        mon = self.sct.monitors[self.monitor_index]
        img = np.array(self.sct.grab(mon))  # BGRA
        return img[:, :, :3]                # BGR


def preprocess(frame_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Convert BGR frame to grayscale and resize to (out_h, out_w)."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (opencv-python) is required for preprocessing but failed to import. " 
            f"Import error: {_cv2_err}"
        )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)
