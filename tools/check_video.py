#!/usr/bin/env python3
"""
영상 진단.

브라우저에서 재생이 안 되거나 스켈레톤이 어긋날 때 원인을 좁힌다.
OpenCV 가 읽은 값과 ffprobe 가 읽은 값을 대조해서 불일치를 찾는다.

  python tools/check_video.py data/videos/*.mp4
  python tools/check_video.py data/videos/user1.mp4 --fix
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from statistics import median
from pathlib import Path

import cv2

# 브라우저가 확실히 디코딩하는 코덱
BROWSER_OK = {"h264", "vp8", "vp9", "av1"}
BROWSER_NO = {"hevc", "h265", "mpeg4", "prores", "vp6"}


def ffprobe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return {"_error": "ffprobe 가 없습니다. ffmpeg 를 설치하세요."}
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
        "-show_streams", "-show_packets", "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames"
        ":packet=pts_time:format=duration",
        str(path),
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        return {"_error": e.stderr.strip()[:200]}
    d = json.loads(raw)
    st = (d.get("streams") or [{}])[0]
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    timestamps = sorted(float(p["pts_time"]) for p in d.get("packets", []) if "pts_time" in p)
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    # Permit timebase rounding; detect actual cadence changes even when r == avg.
    typical = median(intervals) if intervals else 0
    variable = any(abs(delta - typical) > max(.001, typical * .05) for delta in intervals)

    def rate(s):
        try:
            n, den = s.split("/")
            return float(n) / float(den) if float(den) else 0.0
        except Exception:
            return 0.0

    return {
        "codec": st.get("codec_name", "?"),
        "profile": st.get("profile", ""),
        "pix_fmt": st.get("pix_fmt", ""),
        "w": st.get("width"), "h": st.get("height"),
        "r_fps": round(rate(st.get("r_frame_rate", "0/1")), 3),
        "avg_fps": round(rate(st.get("avg_frame_rate", "0/1")), 3),
        "avg_rate": st.get("avg_frame_rate", "0/1"),
        "variable_fps": variable,
        "nb_frames": int(st["nb_frames"]) if st.get("nb_frames") else None,
        "rotation": rot,
        "duration": round(float(d.get("format", {}).get("duration", 0)), 3),
    }


def opencv_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"_error": "OpenCV 가 열지 못했습니다"}
    info = {
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    n = 0
    while cap.grab():
        n += 1
    info["actual"] = n
    cap.release()
    return info


def diagnose(path: Path) -> list:
    fp, cv_ = ffprobe(path), opencv_info(path)
    print(f"\n── {path.name} ──")
    if "_error" in cv_:
        print(f"  OpenCV: {cv_['_error']}")
        return ["OpenCV 가 영상을 열지 못합니다"]
    if "_error" in fp:
        print(f"  ffprobe: {fp['_error']}")
        fp = {}

    print(f"  코덱      {fp.get('codec','?')} {fp.get('profile','')} {fp.get('pix_fmt','')}")
    print(f"  해상도    ffprobe {fp.get('w')}x{fp.get('h')}   OpenCV {cv_['w']}x{cv_['h']}"
          f"   회전 {fp.get('rotation', 0)}도")
    print(f"  프레임레이트  r {fp.get('r_fps')}  avg {fp.get('avg_fps')}  OpenCV {cv_['fps']}")
    print(f"  프레임 수  헤더 {cv_['count']}  실제 {cv_['actual']}  ffprobe {fp.get('nb_frames')}")
    print(f"  길이      {fp.get('duration')}초")

    problems = []
    codec = (fp.get("codec") or "").lower()
    if codec in BROWSER_NO:
        problems.append(f"코덱 {codec} — 브라우저가 디코딩하지 못합니다 (검은 화면의 원인)")
    elif codec and codec not in BROWSER_OK:
        problems.append(f"코덱 {codec} — 브라우저 지원이 불확실합니다")
    if "10le" in (fp.get("pix_fmt") or "") or "High 10" in (fp.get("profile") or ""):
        problems.append("10bit 영상 — 브라우저가 디코딩하지 못할 수 있습니다")

    if fp.get("r_fps") and fp.get("avg_fps"):
        if fp.get("variable_fps") or abs(fp["r_fps"] - fp["avg_fps"]) > 0.2:
            problems.append(
                f"가변 프레임레이트 (r {fp['r_fps']} vs avg {fp['avg_fps']}) — "
                "프레임 번호와 재생 시간이 어긋납니다")
    if abs(cv_["count"] - cv_["actual"]) > 1:
        problems.append(
            f"헤더 프레임 수 {cv_['count']} 와 실제 {cv_['actual']} 불일치 — "
            "라벨 프레임 번호가 틀어질 수 있습니다")
    if fp.get("rotation"):
        problems.append(
            f"회전 메타데이터 {fp['rotation']}도 — 브라우저는 회전해 표시하고 "
            "OpenCV 는 무시해서 스켈레톤이 어긋납니다")
    if fp.get("w") and (fp["w"], fp["h"]) != (cv_["w"], cv_["h"]):
        problems.append(f"해상도 불일치 ffprobe {fp['w']}x{fp['h']} vs OpenCV {cv_['w']}x{cv_['h']}")

    if problems:
        print("  문제")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  문제 없음")
    return problems


def fix(path: Path, outdir: Path, *, fps: float | None = None) -> Path:
    """브라우저·OpenCV 양쪽에서 동일하게 읽히는 형태로 다시 인코딩.
    회전을 픽셀에 굽고, 원래 평균 fps의 CFR, H.264 8bit로 변환한다."""
    probe = ffprobe(path)
    rate = str(fps) if fps is not None else probe.get("avg_rate", str(probe.get("avg_fps", 0)))
    from fractions import Fraction
    if Fraction(rate) <= 0:
        raise ValueError(f"원래 평균 fps를 읽을 수 없습니다: {path}")
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / (path.stem + "_web.mp4")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # 홀수 해상도 방지
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", rate, "-vsync", "cfr",
        "-metadata:s:v", "rotate=0",                  # 회전 메타데이터 제거
        "-movflags", "+faststart", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    print(f"  변환 완료: {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description="영상 진단 및 브라우저 호환 변환")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--fix", action="store_true", help="문제가 있으면 재인코딩")
    ap.add_argument("--fps", type=float, help="변환 fps 지정 (예: 30); 생략하면 원래 평균 fps 유지")
    ap.add_argument("--outdir", type=Path, default=Path("data/videos_web"))
    args = ap.parse_args()

    for v in args.videos:
        if not v.exists():
            print(f"[건너뜀] {v}")
            continue
        problems = diagnose(v)
        if args.fix and problems:
            new = fix(v, args.outdir, fps=args.fps)
            print("  변환본 재진단")
            diagnose(new)
            print(f"\n  라벨 JSON 의 video 를 {new} 로 바꾸고 프레임 번호를 다시 확인하세요.")
            print("  프레임레이트가 바뀌면 라벨 번호도 달라집니다.")


if __name__ == "__main__":
    main()
