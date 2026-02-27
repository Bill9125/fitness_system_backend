import json
import os
import re
import numpy as np
import argparse

def read_skeleton_data(file_path):
    frames = []
    angles = {
        "right_shoulder": [],
        "left_shoulder": [],
        "right_elbow": [],
        "left_elbow": []
    }

    valid_angles = {key: [] for key in angles}  # 儲存有效角度數據

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if not match:
                print(f"❌ Format mismatch: {line.strip()}")
                continue

            frame = int(match.group(1))
            points_str = match.group(2)

            try:
                points_str = points_str.replace("(", "[").replace(")", "]")
                points = eval(f"[{points_str}]")  # 解析成 list
                
                frames.append(frame)
                points = np.array(points)

                # 取出骨架關鍵點
                s0, s1, e0, e1, w0, w1 = points[:6]

                def calculate_angle(v1, v2):
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 == 0 or norm_v2 == 0:
                        return None
                    
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)))
                    return angle  

                # 計算各關節角度
                right_shoulder_angle = 180 - calculate_angle(s1 - s0, e1 - s1)
                left_shoulder_angle = 180 - calculate_angle(s0 - s1, e0 - s0)
                right_elbow_angle = 180 - calculate_angle(e1 - s1, w1 - e1)
                left_elbow_angle = 180 - calculate_angle(e0 - s0, w0 - e0)

                # 收集有效數據
                for key, angle in zip(angles.keys(), 
                                      [right_shoulder_angle, left_shoulder_angle, right_elbow_angle, left_elbow_angle]):
                    if angle is not None:
                        valid_angles[key].append(angle)
                    angles[key].append(angle)  # 仍然存入 None，稍後處理

            except Exception as e:
                for key in angles:
                    angles[key].append(None)

    # 計算各關節的平均值
    avg_angles = {key: (sum(valid_angles[key]) / len(valid_angles[key]) if valid_angles[key] else 0) 
                  for key in angles}

    # 替換 `None` 為該關節的平均值
    for key in angles:
        angles[key] = [angle if angle is not None else avg_angles[key] for angle in angles[key]]

    print(f"✅ Total frames processed: {len(frames)}")
    return frames, angles


def save_json(title, y_label, frames, values, output_path):
    """將計算結果儲存成 JSON"""
    if len(values) > 200:
        trimmed_data = values[100:-100]
    else:
        trimmed_data = values

    # 過濾 None 並確保是 float
    valid_data = [float(v) for v in trimmed_data if v is not None and isinstance(v, (int, float))]

    data = {
        "title": title,
        "y_label": y_label,
        "y_min": 80,
        "y_max": 200,
        "frames": frames,
        "values": values
    }

    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"✅ {title} 已儲存到 {output_path}")

def process_skeleton_data(file_path, output_dir):
    """計算關節角度，並存成 JSON 檔案"""
    frames, angles = read_skeleton_data(file_path)

    # ✅ 儲存 Shoulder_Angle.json（肩膀角度）
    shoulder_values = list(zip(angles["right_shoulder"], angles["left_shoulder"]))
    shoulder_output_path = os.path.join(output_dir, 'Shoulder_Angle.json')
    save_json("Shoulder Joint Angles", "Angle (degrees)", frames, shoulder_values, shoulder_output_path)

    # ✅ 儲存 Elbow_Angle.json（手肘角度）
    elbow_values = list(zip(angles["right_elbow"], angles["left_elbow"]))
    elbow_output_path = os.path.join(output_dir, 'Elbow_Angle.json')
    save_json("Elbow Joint Angles", "Angle (degrees)", frames, elbow_values, elbow_output_path)

# ✅ CLI 參數處理
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "yolo_skeleton_interpolated.txt")
output_dir = os.path.join(out, 'Benchpress_data')

os.makedirs(output_dir, exist_ok=True)
process_skeleton_data(data_path, output_dir)
