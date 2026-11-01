"""Turn an annotated video into a small GIF for the README.

    python scripts/make_gif.py results/demo/annotated.mp4 assets/demo.gif --width 640 --every 2
"""

import argparse

import cv2
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--every", type=int, default=2, help="keep 1 frame out of N")
    p.add_argument("--max-frames", type=int, default=120)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, i = [], 0
    while len(frames) < a.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (a.width, int(h * a.width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    if not frames:
        raise SystemExit("No frames read")
    frames[0].save(a.out, save_all=True, append_images=frames[1:], optimize=True,
                   duration=int(1000 * a.every / fps), loop=0)
    print(f"{len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()
