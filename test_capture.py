import os
from mss import mss, tools

def main():
    print("CWD:", os.getcwd())

    with mss() as sct:
        mon = sct.monitors[1]  # 1 = primary monitor
        shot = sct.grab(mon)

        out_path = os.path.join(os.getcwd(), "test_capture.png")
        tools.to_png(shot.rgb, shot.size, output=out_path)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
