import cv2
import os

def crop_faces_video(input_video, output_video, min_face_size=80):
    """
    从 input_video 中裁剪出最大的人脸区域，生成一个新的 mp4（只有人脸区域）。
    - 不处理音频，只动画面（音频还是用原视频来 load）
    """

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[FaceCrop] Failed to open {input_video}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN check
        fps = 25.0

    # 读取第一帧来估计人脸框
    ret, frame = cap.read()
    if not ret:
        print(f"[FaceCrop] Failed to read first frame from {input_video}")
        cap.release()
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 OpenCV 自带的人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )

    if len(faces) == 0:
        print(f"[FaceCrop] No face found in {input_video}, keep original frame.")
        # 就直接把原视频 copy 一份，不做裁剪
        cap.release()
        # 直接用 ffmpeg copy 也行，这里简单起见就返回 False，让上层用原视频
        return False

    # 取最大的那张脸
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # padding 一点，让裁剪不太贴脸
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    crop_w = x2 - x1
    crop_h = y2 - y1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #os.makedirs(os.path.dirname(output_video), exist_ok=True)
    output_dir = os.path.dirname(output_video)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_video, fourcc, fps, (crop_w, crop_h))

    # 把第一帧写进去
    crop_frame = frame[y1:y2, x1:x2]
    writer.write(crop_frame)

    # 后面所有帧用同一个框裁剪
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop_frame = frame[y1:y2, x1:x2]
        writer.write(crop_frame)

    cap.release()
    writer.release()
    print(f"[FaceCrop] Saved cropped video to {output_video}")
    return True

if __name__ == "__main__":
    # 修改成你自己的测试视频
    input_video = "./Test/chunks/demo_001.mp4"
    output_video = "output_crop.mp4"

    print("=== Testing crop_faces_video() ===")
    print(f"Input : {input_video}")
    print(f"Output: {output_video}")

    success = crop_faces_video(input_video, output_video)

    if success:
        print("Test success! You can open output_crop.mp4 to check the crop result.")
    else:
        print("Test failed or no face detected.")