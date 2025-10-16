import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import argparse

# ==============================
# 1. 参数设置
# ==============================
parser = argparse.ArgumentParser(description="Record RealSense RGB or RGB+Depth video")
parser.add_argument("--camera", choices=["d435i", "405"], default="d435i", help="选择相机型号")
parser.add_argument("--mode", choices=["rgb", "depth"], default="rgb", help="选择录制模式: rgb 或 depth")
parser.add_argument("--width", type=int, default=640, help="图像宽度")
parser.add_argument("--height", type=int, default=480, help="图像高度")
parser.add_argument("--fps", type=int, default=60, help="帧率")
args = parser.parse_args()

# ==============================
# 2. 创建 pipeline 与配置
# ==============================
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色流
config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
# 若选择 depth 模式，则启用深度流
if args.mode == "depth":
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

# 设置输出路径
record_dir = os.path.join(os.getcwd(), "records")
os.makedirs(record_dir, exist_ok=True)
timestamp = time.strftime("%m%d_%H%M%S", time.localtime())
output_file = os.path.join(record_dir, f"{args.camera}_{timestamp}.bag")
config.enable_record_to_file(output_file)


# ==============================
# 3. 启动 pipeline
# ==============================
print(f"[INFO] 开始录制 RealSense {args.mode.upper()} 视频，保存为 {output_file}")
pipeline.start(config)

# ==============================
# 4. 循环录制与显示
# ==============================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if args.mode == "depth" else None

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 若有深度，则可视化显示
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense Depth Recording (press 'q' to stop)", combined)
        else:
            cv2.imshow("RealSense RGB Recording (press 'q' to stop)", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("[INFO] 停止录制，保存文件中...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] 录制完成！")
