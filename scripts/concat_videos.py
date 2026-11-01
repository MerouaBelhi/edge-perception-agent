"""Join several clips (e.g. 8-second AI-generated scenes) into one test video,
normalised to the same size and frame rate.

    python scripts/concat_videos.py assets/scene1.mp4 assets/scene2.mp4 assets/scene3.mp4 \\
        --out assets/site_demo.mp4

Uses the system ffmpeg, or the one shipped with imageio-ffmpeg.
"""

import argparse
import subprocess

from to_h264 import ffmpeg_exe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    a = p.parse_args()

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error"]
    for c in a.clips:
        cmd += ["-i", c]
    norm = "".join(
        f"[{i}:v]scale={a.width}:{a.height}:force_original_aspect_ratio=decrease,"
        f"pad={a.width}:{a.height}:(ow-iw)/2:(oh-ih)/2,fps={a.fps},setsar=1[v{i}];"
        for i in range(len(a.clips)))
    joined = "".join(f"[v{i}]" for i in range(len(a.clips)))
    cmd += ["-filter_complex", f"{norm}{joined}concat=n={len(a.clips)}:v=1:a=0[out]",
            "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    subprocess.run(cmd, check=True)
    print(f"{len(a.clips)} clips -> {a.out}")


if __name__ == "__main__":
    main()
