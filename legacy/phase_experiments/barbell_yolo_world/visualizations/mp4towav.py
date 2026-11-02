from pathlib import Path
import subprocess

mp4_dir = Path("./")

for mp4_file in mp4_dir.glob("*.mp4"):

    output_mp4 = mp4_file.with_stem(mp4_file.stem + "_h264")

    subprocess.run([
        "ffmpeg",
        "-y",                     # 덮어쓰기 확인 안 물어봄
        "-i", str(mp4_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_mp4)
    ], check=True)

    print(f"완료: {output_mp4.name}")
