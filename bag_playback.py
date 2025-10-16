import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import threading
import time
from collections import deque

# ===== 用户配置 =====
file_name = ".bag"
input_file_path = "./records/" + file_name
window_name = "RealSense Smooth Player"
BUFFER_SIZE = 20  # 缓冲帧数量
approx_duration_s = 5 * 60  # 约定总时长（秒）
# ===================

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(input_file_path, repeat_playback=False)
config.enable_stream(rs.stream.color)
pipeline.start(config)

# 获取控制对象
profile = pipeline.get_active_profile()
playback = profile.get_device().as_playback()
playback.set_real_time(True)  # ✅ 让 RealSense 自动以录制帧率播放

fps = 30
try:
    video_profile = profile.get_streams()[0].as_video_stream_profile()
    fps = video_profile.fps()
except Exception:
    pass
frame_time = int(1000 / fps)

# ===== 多线程缓冲读取 =====
frame_buffer = deque(maxlen=BUFFER_SIZE)
stop_flag = False

def reader_thread():
    global stop_flag
    while not stop_flag:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame_buffer.append(np.asanyarray(color_frame.get_data()))
        except RuntimeError:
            stop_flag = True
            break

t = threading.Thread(target=reader_thread, daemon=True)
t.start()

# ===== 播放控制 =====
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar("Seek", window_name, 0, 1000, lambda x: None)

paused = False
seek_target = 0
frame_count = 0

print("✅ 控制说明：空格=暂停/播放，Q=退出，拖动滑块快进。")

try:
    while True:
        if not paused and len(frame_buffer) > 0:
            frame = frame_buffer.popleft()
            frame_count += 1

            # 显示帧
            frame_disp = cv2.resize(frame, (960, 540))
            cv2.putText(frame_disp, f"Frame: {frame_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(window_name, frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            paused = not paused

        # 拖动滑块快进
        slider_val = cv2.getTrackbarPos("Seek", window_name)
        if slider_val != seek_target:
            seek_target = slider_val
            seek_time_ms = int(seek_target / 1000 * approx_duration_s * 1000)
            playback.seek(datetime.timedelta(milliseconds=seek_time_ms))
            frame_buffer.clear()
            paused = False
            time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    stop_flag = True
    t.join(timeout=1.0)
    pipeline.stop()
    cv2.destroyAllWindows()
