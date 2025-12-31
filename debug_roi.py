import cv2, yaml
from capture import MonitorCapture

cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
cap = MonitorCapture(int(cfg.get("monitor_index", 1)))

while True:
    frame = cap.grab()
    x,y,w,h = cfg["roi"]["xp_bar"]
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
    x,y,w,h = cfg["roi"]["hp_bar"]
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow("ROI Debug (ESC to quit)", frame)
    if cv2.waitKey(1) == 27:
        break
