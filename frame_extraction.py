import pyrealsense2 as rs
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

# ========== 配置参数 ==========
file_name = "d435i_rgb_1015_155304"
input_file_path = "/home/wangqin/Applications/project/records/"+file_name+ ".bag"  # .bag 文件路径
output_dir = "./frames/"+file_name
os.makedirs(output_dir, exist_ok=True)

min_interval = 20        # 至少间隔多少帧保存一次
ssim_threshold = 0.8     # 全图结构相似度阈值
diff_threshold = 15.0    # 局部平均像素差异阈值
blur_threshold = 100     # 模糊检测阈值
display_scale = 0.5      # 实时预览缩放比例
gamma_value = 1.2        # Gamma 校正强度
# ============================

# ---------- 工具函数 ----------
def auto_white_balance(img):
    """改进版自动白平衡（避免类型冲突）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])
    lab[:, :, 1] -= ((avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 1.1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image, gamma=1.2):
    """Gamma 校正提亮"""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def is_blurry(image, threshold=100):
    """检测图像是否模糊"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm
# --------------------------------

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(input_file_path, repeat_playback=False)
pipeline.start(config)

profile = pipeline.get_active_profile()
playback = profile.get_device().as_playback()
playback.set_real_time(False)

prev_gray = None
frame_count = 0
saved_count = 0

print("开始抽帧... 按 'q' 退出。\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # === 图像预处理：白平衡 + Gamma ===
        color_image = auto_white_balance(color_image)
        color_image = apply_gamma_correction(color_image, gamma_value)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ssim_val, diff_val = 0, 0
        save_flag = False

        if prev_gray is None:
            save_flag = True
        else:
            # 计算结构相似度与差异
            ssim_val = ssim(gray, prev_gray)
            diff_val = np.mean(cv2.absdiff(gray, prev_gray))

            # 变化大才保存
            if (ssim_val < ssim_threshold or diff_val > diff_threshold) and frame_count % min_interval == 0:
                save_flag = True

        # 模糊检测
        blurry, blur_metric = is_blurry(color_image, threshold=blur_threshold)

        if save_flag and not blurry:
            filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(filename, color_image)
            saved_count += 1
            print(f"[Saved] {filename} | SSIM={ssim_val:.3f} | DIFF={diff_val:.1f} | Sharpness={blur_metric:.1f}")
            prev_gray = gray

        # 实时显示
        preview = cv2.resize(color_image, None, fx=display_scale, fy=display_scale)
        text = f"Frame:{frame_count} | Saved:{saved_count} | SSIM:{ssim_val:.3f} | DIFF:{diff_val:.1f}"
        cv2.putText(preview, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("RealSense Frame Preview", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断。")
            break

        frame_count += 1

except RuntimeError:
    print("播放结束。")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n处理总帧数: {frame_count}")
    print(f"保存图片数: {saved_count}")
    print(f"保存路径: {output_dir}")
