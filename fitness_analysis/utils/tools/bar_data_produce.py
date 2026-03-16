import json, os

def run_bar_data_produce(dir, sport, results):
    # 讀取 yolo 檔案
    yolo_txt_path = os.path.join(
        dir, "coordinates_interpolated.txt")  # 你的 txt 檔案路徑
    
    output_json_path = os.path.join(dir, "config", "Bar_Position.json")  # 輸出的 JSON 檔案
    if sport == 'deadlift':
        idx = 0 # X 中心 (in [x, y] list)
    elif sport == 'benchpress':
        idx = 1 # Y 中心 (in [x, y] list)


    # 初始化數據存儲
    frames = []
    values = []

    # 讀取 YOLO 偵測數據
    for frame_count, parts in results.items():
        center = float(parts[idx])  
        frames.append(frame_count)
        if sport == 'benchpress':
            center = 640 - center
        values.append(center)

    if values:
        x_min = min(values) * 0.9  # X 軸最小值，留 10% 緩衝
        x_max = max(values) * 1.1  # X 軸最大值，留 10% 緩衝
    else:
        x_min = x_max = 0

    # 轉換成 JSON 格式
    data = {
        "title": "Barbell Center Positions",
        "y_label": "Position (pixels)",
        "y_min": round(x_min, 2),
        "y_max": round(x_max, 2),
        "frames": frames,
        "values": [round(v, 2) for v in values]
    }

    # 存成 JSON 檔案
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"✅ JSON 檔案已儲存: {output_json_path}")
