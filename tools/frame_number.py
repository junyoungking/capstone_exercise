#!/usr/bin/env python3
"""
라벨링용 프레임 번호 오버레이.

원본 영상에 프레임 번호 / 초 / 진행 막대를 얹은 사본을 만든다.
이 사본은 눈으로 구간을 읽을 때만 쓰고, 파이프라인에는 원본을 넣는다.

  python tools/frame_number.py data/videos/user_squat_01.mp4
  python tools/frame_number.py data/videos/*.mp4 --slow 2
  python tools/frame_number.py data/videos/user_squat_01.mp4 --grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 220, 60)


def annotate(
    src: Path,
    dst: Path,
    slow: int = 1,
    grid: bool = False,
    tick: int = 10,
) -> None:
    if dst.resolve() == src.resolve():
        raise RuntimeError(
            f"출력 경로가 원본과 같습니다: {src}\n"
            f"  --outdir 을 원본과 다른 폴더로 지정하세요 (기본값: data/numbered)"
        )

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(slow, 1), (w, h)
    )

    # 해상도에 맞춰 글자 크기 조절
    scale = max(w, h) / 900.0
    big = 1.5 * scale
    small = 0.55 * scale
    thick = max(int(3 * scale), 2)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if grid:
            for gx in range(0, w, w // 8):
                cv2.line(frame, (gx, 0), (gx, h), (60, 60, 60), 1)
            for gy in range(0, h, h // 8):
                cv2.line(frame, (0, gy), (w, gy), (60, 60, 60), 1)

        # 프레임 번호 — tick 배수마다 색을 바꿔 스크럽할 때 눈에 띄게
        label = str(i)
        color = CYAN if (tick > 0 and i % tick == 0) else GREEN
        org = (int(18 * scale), int(58 * scale))
        cv2.putText(frame, label, org, FONT, big, BLACK, thick + 4, cv2.LINE_AA)
        cv2.putText(frame, label, org, FONT, big, color, thick, cv2.LINE_AA)

        # 초 표기
        sec = f"{i / fps:6.2f}s"
        org2 = (int(18 * scale), int(88 * scale))
        cv2.putText(frame, sec, org2, FONT, small, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, sec, org2, FONT, small, WHITE, 1, cv2.LINE_AA)

        # 하단 진행 막대
        if total > 0:
            bar_h = max(int(6 * scale), 4)
            cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
            x = int(w * (i + 1) / total)
            cv2.rectangle(frame, (0, h - bar_h), (x, h), color, -1)

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    print(f"{src.name}  {i} 프레임 / {fps:.1f}fps / {w}x{h}  ->  {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="라벨링용 프레임 번호 오버레이")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("data/numbered"))
    ap.add_argument("--slow", type=int, default=1, help="N배 느리게 재생 (fps 를 1/N 로)")
    ap.add_argument("--grid", action="store_true", help="격자 표시")
    ap.add_argument("--tick", type=int, default=10, help="N 배수 프레임 강조, 0 이면 끔")
    args = ap.parse_args()

    for v in args.videos:
        if not v.exists():
            print(f"[건너뜀] 없는 파일: {v}")
            continue
        annotate(v, args.outdir / v.name, args.slow, args.grid, args.tick)


if __name__ == "__main__":
    main()