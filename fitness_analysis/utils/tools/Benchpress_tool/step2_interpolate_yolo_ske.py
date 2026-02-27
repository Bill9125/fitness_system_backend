import numpy as np
import re
import os
from tqdm import tqdm
from scipy.interpolate import interp1d

def read_bar_frames(file_path):
    frames = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].isdigit():
                frames.append(int(parts[0]))
    return sorted(frames)

def read_skeleton_data_with_nan(file_path):
    skeleton_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if match:
                frame = int(match.group(1))
                points_str = match.group(2).replace("(", "[").replace(")", "]")
                try:
                    points = eval(f"[{points_str.replace('NaN', 'np.nan')}]")
                    skeleton_data[frame] = np.array(points)
                except Exception as e:
                    print(f"❌ Skipping frame {frame}: {e}")
    return skeleton_data

def interpolate_skeleton_with_nan(bar_frames, skeleton_data, num_keypoints):
    interpolated_data = {}
    keypoint_x = {i: [] for i in range(num_keypoints)}
    keypoint_y = {i: [] for i in range(num_keypoints)}
    valid_frames = sorted(skeleton_data.keys())

    for frame in valid_frames:
        points = skeleton_data[frame]
        for i in range(num_keypoints):
            x, y = points[i]
            keypoint_x[i].append((frame, x))
            keypoint_y[i].append((frame, y))

    interp_x_func = {}
    interp_y_func = {}

    for i in range(num_keypoints):
        fx = np.array([f for f, v in keypoint_x[i] if not np.isnan(v)])
        vx = np.array([v for f, v in keypoint_x[i] if not np.isnan(v)])
        fy = np.array([f for f, v in keypoint_y[i] if not np.isnan(v)])
        vy = np.array([v for f, v in keypoint_y[i] if not np.isnan(v)])

        interp_x_func[i] = interp1d(fx, vx, kind='linear', fill_value='extrapolate') if len(fx) > 1 else None
        interp_y_func[i] = interp1d(fy, vy, kind='linear', fill_value='extrapolate') if len(fy) > 1 else None

    for frame in bar_frames:
        points = []
        for i in range(num_keypoints):
            x = float(interp_x_func[i](frame)) if interp_x_func[i] is not None else np.nan
            y = float(interp_y_func[i](frame)) if interp_y_func[i] is not None else np.nan
            points.append((x, y))
        interpolated_data[frame] = points

    return interpolated_data

def write_interpolated_skeleton(output_path, interpolated_data):
    with open(output_path, 'w') as f:
        for frame, points in sorted(interpolated_data.items()):
            points_str = ", ".join([f"({p[0]}, {p[1]})" for p in points])
            f.write(f"Frame {frame}: [[{points_str}]]\n")

def process_subject_folder(folder):
    bar_file = os.path.join(folder, 'yolo_coordinates_interpolated_hampel.txt')
    if not os.path.exists(bar_file):
        return

    tasks = [
        ('yolo_skeleton_top_11m_hampel.txt', 'yolo_skeleton_top_11m_interpolated_hampel.txt', 8),
        ('yolo_skeleton_hampel.txt', 'yolo_skeleton_interpolated_hampel.txt', 6)
    ]

    for input_file, output_file, num_kpts in tasks:
        input_path = os.path.join(folder, input_file)
        output_path = os.path.join(folder, output_file)

        if not os.path.exists(input_path):
            continue
        if os.path.exists(output_path):
            continue

        try:
            bar_frames = read_bar_frames(bar_file)
            skeleton_data = read_skeleton_data_with_nan(input_path)
            if not skeleton_data:
                continue
            print(f"✅ [{input_file}] 讀入 {len(skeleton_data)} 幀")
            interpolated_data = interpolate_skeleton_with_nan(bar_frames, skeleton_data, num_kpts)
            write_interpolated_skeleton(output_path, interpolated_data)
        except Exception as e:
            print(f"⚠️ 錯誤處理 {folder} 中的 {input_file}: {e}")

def process_all_folders(base_root):
    for category in os.listdir(base_root):
        category_path = os.path.join(base_root, category)
        if not os.path.isdir(category_path):
            continue
        for subject in tqdm(os.listdir(category_path), desc=f"處理類別：{category}"):
            subject_path = os.path.join(category_path, subject)
            if os.path.isdir(subject_path):
                process_subject_folder(subject_path)


if __name__ == "__main__":                                                                                       # 程式入口
    import sys, os                                                                                               # 匯入模組
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                     # 預設 recordings 根目錄
    if len(sys.argv) >= 2:                                                                                       # 有傳入參數
        base_path = sys.argv[1]                                                                                  # 指定目標資料夾
        if not os.path.isdir(base_path):                                                                         # 檢查有效性
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                          # 拋錯
    else:                                                                                                        # 無參數 → 退回找最新
        if not os.path.isdir(recordings_dir):                                                                    # 檢查根目錄
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                              # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)                              # 列子資料夾
                if os.path.isdir(os.path.join(recordings_dir, d))]                                               # 過濾
        if not subs:                                                                                             # 沒有子資料夾
            raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾，且未提供參數")                         # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                             # 取最新資料夾
    print(f"interpolate ske process folder : {base_path}")                                                       # 顯示實際處理資料夾（骨架內插）
    process_subject_folder(base_path)                                                                            # 執行主流程
    print("✅ 所有 yolo_skeleton 檔案已補齊內插（如尚未存在）")                                                      # 完成訊息

