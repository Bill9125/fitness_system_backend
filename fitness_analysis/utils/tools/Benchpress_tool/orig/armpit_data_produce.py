import json, os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "interpolated_mediapipe_landmarks_1.txt")
outpu_path = os.path.join(out, 'Benchpress_data', 'Armpit_Angle.json')

# ✅ 計算關節角度
def calculate_joint_angle(p1, p2, p3):
    """計算關節角度，p1-p2-p3 為三個點"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    # 計算夾角
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    # 如果 angle 是 NaN，返回 None
    return None if np.isnan(angle) else angle

# 初始化數據存儲
frames = []
values = []

# 解析 YOLO txt 檔案
landmarks = {}  # { frame: { landmark_id: (x, y) } }

with open(data_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        
        frame = int(parts[0])
        landmark_id = int(parts[1])
        x, y = float(parts[2]), float(parts[3])

        if frame not in landmarks:
            landmarks[frame] = {}
        landmarks[frame][landmark_id] = (x, y)

# ✅ 計算每一幀的角度
all_valid_angles = []

for frame, points in landmarks.items():
    if all(k in points for k in [11, 12, 13, 14, 23, 24]):  # 確保數據完整
        left_angle = calculate_joint_angle(points[13], points[11], points[23])  # 13-11-23
        right_angle = calculate_joint_angle(points[14], points[12], points[24])  # 14-12-24

        # 儲存所有有效角度
        for angle in [left_angle, right_angle]:
            if angle is not None:
                all_valid_angles.append(angle)

        frames.append(frame)
        values.append((left_angle, right_angle))

# ✅ 計算有效角度的平均值
if all_valid_angles:
    avg_angle = sum(all_valid_angles) / len(all_valid_angles)
else:
    avg_angle = 0  # 若沒有有效數據，則預設為 0

# ✅ 替換 None 為平均值
values = [[v if v is not None else avg_angle for v in pair] for pair in values]

# ✅ 計算 y_min 和 y_max
valid_values = [angle for pair in values for angle in pair]  # 這時已經沒有 None

if valid_values:
    y_min = min(valid_values) * 0.9  # 讓範圍多 10%
    y_max = max(valid_values) * 1.1
else:
    y_min = y_max = 0

# ✅ 轉換成 JSON 格式
data = {
    "title": "Armpit Angles",
    "y_label": "Angle (degrees)",
    "y_min": y_min,
    "y_max": y_max,
    "frames": frames,
    "values": values  # 此時已無 None
}

# ✅ 存成 JSON 檔案
with open(outpu_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"✅ JSON 檔案已儲存: {outpu_path}")
