import os
import cv2
import imagehash
from PIL import Image

# ==============================
# 参数设置
# ==============================
show_scale = 0.7        # 预览缩放比例
threshold = 10          # pHash差异阈值（越小越严格）
input_dir = './records/phone/videos'
input_file = ''         # 指定单个视频文件（留空则批量）
output_dir = './keyframes/phone_phash'

# ==============================
# 视频列表
# ==============================
if input_file:
    video_files = [input_file]
else:
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

print(f"检测到 {len(video_files)} 个视频：{video_files}")

# ==============================
# 主循环
# ==============================
for video_name in video_files:
    input_path = os.path.join(input_dir, video_name)
    output_path = os.path.join(output_dir, os.path.splitext(video_name)[0])
    os.makedirs(output_path, exist_ok=True)

    print(f"开始处理：{video_name}")
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        print(f"无法读取视频：{video_name}")
        cap.release()
        continue

    # 初始化pHash
    prev_hash = imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 当前帧的pHash
        cur_hash = imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        diff = prev_hash - cur_hash

        # 判断是否为关键帧
        isKeyframe = diff > threshold
        if isKeyframe:
            filename = os.path.join(output_path, f"keyframe{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            prev_hash = cur_hash

        # ==============================
        # 回放预览部分
        # ==============================
        display = frame.copy()
        h, w = display.shape[:2]
        display = cv2.resize(display, (int(w * show_scale), int(h * show_scale)))

        # 绘制半透明信息框
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (720, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        # 绘制状态文本
        color = (0, 255, 0) if isKeyframe else (0, 0, 255)
        text = f"{video_name} | Frame: {frame_idx}, Diff: {diff:.2f} {'Saved' if isKeyframe else ''}"
        cv2.putText(display, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("pHash-frame extraction, press 'p' to exit ", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"✅ {video_name} 抽帧完成，共保存 {saved} 张关键帧 -> {output_path}")

cv2.destroyAllWindows()
