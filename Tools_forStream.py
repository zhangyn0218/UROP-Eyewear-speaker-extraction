import os, glob, time, threading, queue, subprocess, tempfile
import cv2
import torch
import soundfile as sf
import sounddevice as sd
from model import SE
from tools import load_audio, load_visual, preprocess, save_results
from Face_croping import *

# ================== 0. 全局配置 ==================

BASE_NAME       = "demo"
CHUNKS_DIR      = "Test/chunks"
EXPS_DIR        = "Test/chunksExp"
CHUNK_DURATION  = 0.6
BUFFER_SECONDS  = 0.5


output_queue = queue.Queue()


producer_done = False
producer_done_lock = threading.Lock()
stream_start_time = time.time()
producer_start_time = None
consumer_first_play_time = None


#model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = SE().to(device)
model.eval()


def process_one_video(input_video_name, out_video_name, out_audio_name, mix_audio_name):

    base, ext = os.path.splitext(input_video_name)
    cropped_video_name = base + "_crop" + ext

    use_crop = crop_faces_video(input_video_name, cropped_video_name)
    if use_crop:
        visual_video_name = cropped_video_name
    else:
        # 回退：如果裁剪失败/没找到脸，就用原视频
        visual_video_name = input_video_name
    # Step 1: Load the audio
    mix_audio = load_audio(input_video_name, mix_audio_name)
    # Step 2: Load the visual
    faces = load_visual(input_video_name)

    # Step 3: Preprocess, align the length
    mix_audio, faces = preprocess(mix_audio, faces)

    mix_audio = mix_audio.to(device)
    faces = faces.to(device)

    # Step 4: Evaluate
    with torch.no_grad():
        out = model(mix_audio, faces)[0]

    # Step 5: Save the results
    save_results(input_video_name, out_audio_name, out_video_name, out)
    print("finished")

#Live version, will wait for new chunk coming in
def producer_thread_live():

    global producer_done, producer_start_time

    os.makedirs(EXPS_DIR, exist_ok=True)

    idx = 0
    print("[Producer] Live mode started.")

    while True:
        expected_chunk = os.path.join(CHUNKS_DIR, f"{BASE_NAME}_{idx:03d}.mp4")
        print(f"[Producer] Waiting for chunk: {expected_chunk}")


        wait_start = time.time()

        #wair until time exceed expect time
        while not os.path.exists(expected_chunk):
            elapsed = time.time() - wait_start

            if elapsed > 2 * CHUNK_DURATION:
                print(f"[Producer] Timeout waiting for {expected_chunk} "
                      f"({elapsed:.2f}s > {2 * CHUNK_DURATION:.2f}s). "
                      "Assume no more chunks, exiting producer.")
                with producer_done_lock:
                    producer_done = True
                return

            time.sleep(0.1)

        in_video = expected_chunk
        tag = f"{BASE_NAME}_{idx:03d}"

        out_video_name = os.path.join(EXPS_DIR, f"{tag}_res.mp4")
        out_audio_name = os.path.join(EXPS_DIR, f"{tag}_res.wav")
        mix_audio_name = os.path.join(EXPS_DIR, f"{tag}.wav")

        print(f"[Producer] Processing chunk {idx}: {in_video}")

        if producer_start_time is None:
            producer_start_time = time.time()

        chunk_start = time.time()
        process_one_video(in_video, out_video_name, out_audio_name, mix_audio_name)
        proc_time = time.time() - chunk_start
        print(f"[Producer] Done {idx}, process_time = {proc_time:.2f}s")

        #Simulate real chunk duration
        if proc_time < CHUNK_DURATION:
            sleep_time = CHUNK_DURATION - proc_time
            print(f"[Producer] Sleeping {sleep_time:.2f}s to simulate real-time")
            time.sleep(sleep_time)
        else:
            print(f"[Producer] Slower than real-time by {proc_time - CHUNK_DURATION:.2f}s")

        output_queue.put((out_video_name, out_audio_name))

        idx += 1

#Not live -> new chunks won't be operated.
def producer_thread():
    global producer_done, producer_start_time

    os.makedirs(EXPS_DIR, exist_ok=True)
    pattern = os.path.join(CHUNKS_DIR, f"{BASE_NAME}_*.mp4")
    chunk_files = sorted(glob.glob(pattern))
    print(f"[Producer] Found {len(chunk_files)} chunks")

    for idx, in_video in enumerate(chunk_files):
        tag = f"{BASE_NAME}_{idx:03d}"

        out_video_name = os.path.join(EXPS_DIR, f"{tag}_res.mp4")
        out_audio_name = os.path.join(EXPS_DIR, f"{tag}_res.wav")
        mix_audio_name = os.path.join(EXPS_DIR, f"{tag}.wav")

        print(f"[Producer] Processing chunk {idx}: {in_video}")

        if producer_start_time is None:
            producer_start_time = time.time()

        chunk_start = time.time()
        process_one_video(in_video, out_video_name, out_audio_name, mix_audio_name)
        proc_time = time.time() - chunk_start
        print(f"[Producer] Done {idx}, process_time = {proc_time:.2f}s")


        if proc_time < CHUNK_DURATION:
            sleep_time = CHUNK_DURATION - proc_time
            print(f"[Producer] Sleeping {sleep_time:.2f}s to simulate real-time chunk arrival")
            time.sleep(sleep_time)
        else:
            print(f"[Producer] Processing slower than real-time by {proc_time - CHUNK_DURATION:.2f}s")


        output_queue.put(out_video_name)


    with producer_done_lock:
        producer_done = True
    print("[Producer] All chunks processed. Producer done.")


def simulate_stream_from_chunks(base_name,
                                chunks_dir="Test/chunks",
                                exps_dir="Test/exps",
                                chunk_duration=2.0,
                                ideal_buffer_time=2.0):

    os.makedirs(exps_dir, exist_ok=True)

    pattern = os.path.join(chunks_dir, f"{base_name}_*.mp4")
    chunk_files = sorted(glob.glob(pattern))
    print(f"Found {len(chunk_files)} chunks")

    total_input_time = 0.0
    total_process_time = 0.0
    cum_lag = 0.0
    max_lag = 0.0

    out_chunk_videos = []

    for idx, in_video in enumerate(chunk_files):
        tag = f"{base_name}_{idx:03d}"

        out_video_name = os.path.join(exps_dir, f"{tag}_res.mp4")
        out_audio_name = os.path.join(exps_dir, f"{tag}_res.wav")
        mix_audio_name = os.path.join(exps_dir, f"{tag}.wav")

        print(f"\n--- Chunk {idx}: {in_video} ---")
        t0 = time.time()
        process_one_video(in_video, out_video_name, out_audio_name, mix_audio_name)
        t1 = time.time()

        proc_time = t1 - t0
        total_process_time += proc_time
        total_input_time += chunk_duration

        lag_delta = proc_time - chunk_duration
        cum_lag += lag_delta
        max_lag = max(max_lag, cum_lag)

        print(f"chunk_duration = {chunk_duration:.2f}s, "
              f"proc_time = {proc_time:.2f}s, "
              f"cum_lag = {cum_lag:.2f}s")

        out_chunk_videos.append(out_video_name)

    print("\n=== Stream summary ===")
    print(f"Total input time   : {total_input_time:.2f}s")
    print(f"Total process time : {total_process_time:.2f}s")
    print(f"Max lag (worst)    : {max_lag:.2f}s")
    print(f"Ideal buffer time  : {ideal_buffer_time:.2f}s")

    return out_chunk_videos, max_lag


#ffplay
def feed_chunk_to_ffplay(chunk_video, ffplay_stdin):

    if isinstance(chunk_video, tuple):
        chunk_video = chunk_video[0]
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", chunk_video,
        "-f", "mpegts",
        "-codec:v", "copy",
        "-codec:a", "copy",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            ffplay_stdin.write(data)
    except BrokenPipeError:
        pass
    finally:
        proc.stdout.close()
        proc.wait()

# sounddevice -> sounds, opencv -> visuals
def play_chunk(video_path, audio_path, window_name="Stream Output"):

    try:
        audio_data, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"[Consumer] Failed to read audio {audio_path}: {e}")
        audio_data, sr = None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Consumer] Failed to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN
        fps = 25.0
    frame_interval = 1.0 / fps

    if audio_data is not None and sr is not None:
        sd.play(audio_data, sr, blocking=False)

    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)
        now = time.time()
        sleep_time = frame_interval - (now - last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_time = time.time()

        key = cv2.waitKey(1) & 0xFF

    cap.release()
    sd.stop()

#use OpenCV to output Visual ONLY
def play_video_file(video_path, window_name="Stream Output"):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Consumer] Failed to open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    delay = int(1000 / fps)

    print(f"[Consumer] Playing {video_path} at {fps:.2f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF

    cap.release()
    return False


def consumer_thread():
    global consumer_first_play_time
    window_name = "Stream Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    buffered_chunks = []
    buffered_time = 0.0
    started_playback = False
    # ffplay_proc = None

    print("[Consumer] Started. Waiting for buffer...")

    while True:
        if not started_playback:
            try:
                video_path = output_queue.get(timeout=0.5)
            except queue.Empty:
                with producer_done_lock:
                    done = producer_done
                if done and output_queue.empty():
                    print("[Consumer] Producer done and queue empty. Nothing to play.")
                    break
                continue

            buffered_chunks.append(video_path)
            buffered_time += CHUNK_DURATION
            print(f"[Consumer] Buffered {len(buffered_chunks)} chunks, time = {buffered_time:.2f}s")


            if buffered_time >= BUFFER_SECONDS:
                ffplay_proc = subprocess.Popen(
                    ["ffplay", "-autoexit", "-loglevel", "error", "-"],
                    stdin=subprocess.PIPE
                )
                consumer_first_play_time = time.time()
                print("[Consumer] Buffer ready. Start playback!")
                if producer_start_time is not None:
                    delay = consumer_first_play_time - producer_start_time
                    print(f"：Startup_latency: {delay:.2f} s")

                started_playback = True

        else:
            if not buffered_chunks:
                try:
                    video_path = output_queue.get(timeout=0.5)
                except queue.Empty:
                    with producer_done_lock:
                        done = producer_done
                    if done and output_queue.empty():
                        print("[Consumer] No more chunks to play. Exiting.")
                        break
                    continue
                buffered_chunks.append(video_path)
                buffered_time += CHUNK_DURATION

            video_path, audio_path = buffered_chunks.pop(0)
            buffered_time -= CHUNK_DURATION

            ''' #cv2 only
            user_quit = play_video_file(current_video, window_name=window_name)
            if user_quit:
                print("[Consumer] User pressed 'q'. Stop playback.")
                break
            '''
             #opencv+soundfile, no system package support
            try:
                play_chunk(video_path, audio_path, window_name=window_name)
            except KeyboardInterrupt:
                print("[Consumer] User pressed 'q'. Stop playback.")
                break
            '''
            try:
                feed_chunk_to_ffplay(current_video, ffplay_proc.stdin)
            except BrokenPipeError:
                print("[Consumer] ffplay closed. Stop playback.")
                break
            '''

    # cv2.destroyAllWindows()
    print("[Consumer] Exit.")


def concat_all_output_chunks_to_single_video():

    pattern = os.path.join(EXPS_DIR, f"{BASE_NAME}_*_res.mp4")
    out_videos = sorted(glob.glob(pattern))
    if not out_videos:
        print("[Concat] No output videos found.")
        return

    final_output = os.path.join(EXPS_DIR, f"{BASE_NAME}_stream_res.mp4")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        list_file = f.name
        for v in out_videos:
            f.write(f"file '{os.path.abspath(v)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        final_output
    ]
    print("[Concat] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[Concat] Saved to:", final_output)



