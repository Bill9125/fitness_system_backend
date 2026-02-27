import json, os
import argparse
import sys

# 設定參數解析
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--sport', type=str)
args = parser.parse_args()

target_dir = args.dir
# out = args.out  # 目前看起來沒用到，保留
sport = args.sport

# [修正 1] 統一轉為小寫，避免 'Squat' vs 'squat' 造成判斷錯誤
sport = sport.lower() if sport else ""

# 讀取 yolo 檔案
yolo_txt_path = os.path.join(target_dir, "yolo_coordinates_interpolated.txt")

# 檢查輸入檔案是否存在
if not os.path.exists(yolo_txt_path):
    print(f"[Error] 找不到座標檔案: {yolo_txt_path}")
    sys.exit(1)

# [修正 2] 設定輸出路徑，並增加防呆預設值
output_json_path = None

if sport == 'deadlift':
    output_json_path = os.path.join(target_dir, "Bar_Position.json")
elif sport == 'benchpress':
    output_json_path = os.path.join(target_dir, "Bar_Position.json")
elif sport == 'squat':  # [修正] 改為小寫判斷
    output_json_path = os.path.join(target_dir, "Bar_Position.json")
else:
    print(f"[Error] 未知的運動類型: {sport}")
    sys.exit(1)

# 初始化數據存儲
frames = []
values = []

# 讀取 YOLO 偵測數據
# 假設格式為: frame_index, x_center, y_center, ...
with open(yolo_txt_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        
        # 確保資料長度足夠 (至少要有 frame, x, y)
        if len(parts) < 3:
            continue
        
        try:
            frame_count = int(parts[0])     # 幀數
            
            # [修正 3] 改用 Y 座標 (parts[2]) 來顯示垂直軌跡
            # 原本是 float(parts[1]) -> X 軸 (左右偏移)
            # 改為 float(parts[2]) -> Y 軸 (上下高度)
            # 注意：在影像座標中，Y 越大代表越下面。如果圖表上下顛倒，可考慮用 (影像高 - y)
            y_center = float(parts[2])      
            
            frames.append(frame_count)
            values.append(y_center)
        except ValueError:
            continue

if not values:
    print(f"[Warning] {yolo_txt_path} 內無有效數據")
    sys.exit(0)

# 計算 Y 軸範圍 (原本變數叫 x_min/x_max，這裡改名比較不混淆，但邏輯不變)
y_min_val = min(values)
y_max_val = max(values)
range_val = y_max_val - y_min_val
if range_val == 0: range_val = 1  # 避免除以零或範圍過小

# 上下各留 10% 緩衝
plot_min = y_min_val - (range_val * 0.1)
plot_max = y_max_val + (range_val * 0.1)

# 轉換成 JSON 格式
data = {
    "title": "Barbell Vertical Trajectory", # 修改標題更貼切
    "y_label": "Position (y-pixels)",
    "y_min": plot_min,
    "y_max": plot_max,
    "frames": frames,
    "values": values
}

# 存成 JSON 檔案
try:
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"✅ JSON 檔案已儲存: {output_json_path}")
    print(f"   (Sport: {sport}, Frames: {len(frames)})")
except Exception as e:
    print(f"[Error] 寫入 JSON 失敗: {e}")