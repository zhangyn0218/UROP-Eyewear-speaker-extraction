import os
import subprocess

def split_video_into_chunks(input_video, out_dir, chunk_duration, prefix="chunk"):
    """
    使用 ffmpeg 将输入视频切成多个 chunk，每个 chunk 为 chunk_duration 秒。

    input_video: 输入视频路径
    out_dir: 输出目录，比如 "chunks/"
    chunk_duration: 每段的秒数，例如 2.0
    prefix: 输出文件名前缀，比如 chunk，则输出 chunk_000.mp4, chunk_001.mp4...
    """
    os.makedirs(out_dir, exist_ok=True)

    # 输出模式：chunk_000.mp4, chunk_001.mp4 ...
    output_pattern = os.path.join(out_dir, f"{prefix}_%03d.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-c", "copy",                  # 不重新编码，速度快
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        output_pattern
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 返回切好的chunk路径
    chunks = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith(prefix) and f.endswith(".mp4"))
    return chunks


# ========== 使用示例 ==========
if __name__ == "__main__":
    input_video = "./data/test2.mp4"
    chunks = split_video_into_chunks(
        input_video=input_video,
        out_dir="./chunks",
        chunk_duration=2.0,
        prefix="demo"
    )

    print("Finish")
    for c in chunks:
        print(c)
