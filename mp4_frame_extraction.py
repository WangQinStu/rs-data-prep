import os
import cv2


show_scale = 0.7
threshold = 0.8

input_dir = "./records/phone/videos"
input_file = "/VID_20251017_101602.mp4"
input_path = input_dir + input_file

output_path = "./keyframes/phone" + input_file
os.makedirs(output_path,exist_ok=True)

cap = cv2.VideoCapture(input_path)
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("无法读取视频")

#初始化直方图
prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame,cv2.COLOR_BGR2HSV)], [0,1], None, [50,50], [0,180,0,256])
cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)

frame_idx = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [50,50], [0,180,0,256])
    cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
    #计算相似度 并 根据相似度保存图片
    similarity = cv2.compareHist(prev_hist,hist,cv2.HISTCMP_CORREL)
    isKeyframe = similarity < threshold
    if isKeyframe:
        filename = os.path.join(output_path, f"keyframe_{frame_idx:05d}.jpg")
        cv2.imwrite(filename,frame)
        saved += 1
        prev_hist = hist

    # ----------回放预览----------
    display = frame.copy()
    h, w = display.shape[:2]
    display = cv2.resize(display, (int(w * show_scale), int(h * show_scale)))


    # 画个半透明背景条，增强可读性
    # overlay = display.copy()
    # cv2.rectangle(overlay, (10, 10), (700, 60), (0, 0, 0), -1)
    # cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

    # 绘制文字
    color = (0, 255, 0) if isKeyframe else (0, 0, 255)
    text = f"Frame: {frame_idx}, Sim: {similarity:.3f} {'[SAVE]' if isKeyframe else ''}"
    cv2.putText(display, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("playback(press 'q' to exit)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ 抽帧完成，共保存 {saved} 张关键帧至 {output_path}/")
