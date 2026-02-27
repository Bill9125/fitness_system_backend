import numpy as np
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()
dir = args.dir
####處理一部影片的
def read_bar_frames(file_path):
    """ 讀取 new_bar_interpolate.txt 獲取完整的幀數列表 """
    frames = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frames.append(int(parts[0]))  # 取第一欄（幀數）
    return sorted(frames)

def read_skeleton_data(file_path):
    """ 讀取 vision3_new_skeleton.txt 解析骨架數據 """
    skeleton_data = {}  # 存放幀數對應的骨架座標
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if match:
                frame = int(match.group(1))
                points_str = match.group(2).replace("(", "[").replace(")", "]")
                try:
                    points = eval(f"[{points_str}]")  # 解析為 Python list
                    skeleton_data[frame] = np.array(points)  # 存入對應幀數
                except Exception as e:
                    print(f"Skipping frame {frame} due to error: {e}")

    return skeleton_data

def interpolate_skeleton(bar_frames, skeleton_data):
    """ 根據 bar_frames 進行線性內插補足缺失的骨架數據 """
    all_frames = sorted(skeleton_data.keys())
    interpolated_data = {}

    for target_frame in bar_frames:
        if target_frame in skeleton_data:
            interpolated_data[target_frame] = skeleton_data[target_frame]
        else:
            # 找到最接近的前後幀
            prev_frame = max([f for f in all_frames if f < target_frame], default=None)
            next_frame = min([f for f in all_frames if f > target_frame], default=None)

            if prev_frame is not None and next_frame is not None:
                # 進行線性內插
                weight = (target_frame - prev_frame) / (next_frame - prev_frame)
                interpolated_data[target_frame] = (1 - weight) * skeleton_data[prev_frame] + weight * skeleton_data[next_frame]
            elif prev_frame is not None:
                # 只有前一幀數據，複製前一幀
                interpolated_data[target_frame] = skeleton_data[prev_frame]
            elif next_frame is not None:
                # 只有後一幀數據，複製後一幀
                interpolated_data[target_frame] = skeleton_data[next_frame]
            else:
                # 沒有可用的前後幀數據（不應該發生）
                interpolated_data[target_frame] = np.full_like(skeleton_data[all_frames[0]], np.nan)

    return interpolated_data

def write_interpolated_skeleton(output_path, interpolated_data):
    """ 將補全的骨架數據寫入新的檔案 """
    with open(output_path, 'w') as f:
        for frame, points in sorted(interpolated_data.items()):
            points_str = ", ".join([f"({p[0]}, {p[1]})" for p in points])
            f.write(f"Frame {frame}: [[{points_str}]]\n")

# 設定檔案路徑
bar_file = os.path.join(dir, 'yolo_coordinates_interpolated.txt')
skeleton_file = os.path.join(dir, 'yolo_skeleton.txt')
output_file = os.path.join(dir, 'yolo_skeleton_interpolated.txt')

# 讀取數據
bar_frames = read_bar_frames(bar_file)
skeleton_data = read_skeleton_data(skeleton_file)

# 進行線性內插
interpolated_data = interpolate_skeleton(bar_frames, skeleton_data)

# 輸出補全的數據
write_interpolated_skeleton(output_file, interpolated_data)

print(f"✅ 補全的骨架數據已儲存至 {output_file}")


