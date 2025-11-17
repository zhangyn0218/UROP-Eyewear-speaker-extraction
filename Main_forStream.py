from Tools_forStream import *


BASE_NAME       = "demo"        # 原视频名 + chunk 前缀，比如 test2_000.mp4
CHUNKS_DIR      = "Test/chunks"  # 你已有的 chunk 目录
EXPS_DIR        = "Test/chunksExp"    # 输出目录（和原来一致）
CHUNK_DURATION  = 0.6            # 每个 chunk 的长度（秒）
BUFFER_SECONDS  = 0.5           # 期望的播放延迟（buffer）

if __name__ == "__main__":

    # 启动 producer 和 consumer 两个线程
    t_producer = threading.Thread(target=producer_thread_live, daemon=True)
    # t_producer = threading.Thread(target=producer_thread, daemon=True)
    t_consumer = threading.Thread(target=consumer_thread, daemon=False)

    t_producer.start()
    t_consumer.start()

    # 等待 consumer 播放结束
    t_consumer.join()
    print("[Main] Streaming finished.")


    # 如需最后生成完整结果视频，可以再调用：
    concat_all_output_chunks_to_single_video()