"""Convert the annotated video to H.264 MP4, the format LinkedIn, browsers and
phones play (OpenCV writes 'mp4v', which many players reject).

    python scripts/to_h264.py results/site/annotated.mp4 results/site/linkedin.mp4
    python scripts/to_h264.py results/site/annotated.mp4 out.mp4 --start 5 --duration 20
    python scripts/to_h264.py assets/slow.mp4 assets/site.mp4 --speed 2   # slow-motion stock clip

Uses the system ffmpeg if there is one, otherwise the binary shipped with the
imageio-ffmpeg package (pip install imageio-ffmpeg). No root rights needed.
"""

import argparse
import shutil
import subprocess


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: run  pip install imageio-ffmpeg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--start", type=float, default=None, help="seconds to skip")
    p.add_argument("--duration", type=float, default=None, help="seconds to keep")
    p.add_argument("--max-width", type=int, default=1920)
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed factor, e.g. 2 = twice as fast (for slow-motion clips)")
    a = p.parse_args()

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error"]
    if a.start is not None:
        cmd += ["-ss", str(a.start)]
    cmd += ["-i", a.src]
    if a.duration is not None:
        cmd += ["-t", str(a.duration)]
    # even dimensions (required by yuv420p), capped width, broad player compatibility
    vf = f"scale='min({a.max_width},iw)':-2"
    if a.speed != 1.0:
        vf = f"setpts=PTS/{a.speed}," + vf
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", a.dst]
    subprocess.run(cmd, check=True)
    print(f"-> {a.dst}")


if __name__ == "__main__":
    main()
