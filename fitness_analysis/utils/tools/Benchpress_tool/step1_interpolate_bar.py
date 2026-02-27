import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_yolo(yolo_data):
    if len(yolo_data) == 0:
        return yolo_data

    frames = yolo_data[:, 0]
    interpolated_data = [frames]

    for col in range(1, 5):
        values = yolo_data[:, col]
        valid = ~np.isnan(values)
        if valid.sum() >= 2:
            interp_func = interp1d(frames[valid], values[valid], kind='linear', fill_value='extrapolate')
            interpolated_values = interp_func(frames)
        else:
            interpolated_values = np.nan_to_num(values)
        interpolated_data.append(interpolated_values)

    return np.stack(interpolated_data, axis=1)

def load_yolo_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) == 5:
                try:
                    row = [float(v) if v.strip() != '' else np.nan for v in values]
                    data.append(row)
                except:
                    continue
    return np.array(data)

def save_yolo_output(folder, yolo_interp_data):
    np.savetxt(
        os.path.join(folder, 'yolo_coordinates_interpolated_hampel.txt'),
        yolo_interp_data,
        delimiter=',',
        fmt='%d,%.8f,%.8f,%.8f,%.8f'
    )

def process_subject_folder(folder):
    yolo_file = os.path.join(folder, 'yolo_coordinates_hampel.txt')
    if not os.path.exists(yolo_file):
        return

    try:
        yolo_raw = load_yolo_data(yolo_file)
        if yolo_raw.shape[0] == 0:
            return

        yolo_interp = interpolate_yolo(yolo_raw)
        save_yolo_output(folder, yolo_interp)
    except Exception as e:
        print(f"⚠️ 錯誤處理 {folder}: {e}")

def process_all_folders(base_root):
    if not os.path.exists(base_root):
        print(f"❌ 找不到資料夾: {base_root}")
        return

    for category in os.listdir(base_root):
        category_path = os.path.join(base_root, category)
        if not os.path.isdir(category_path):
            continue
        for subject in tqdm(os.listdir(category_path), desc=f"處理 {category}"):
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
    print(f"interpolate bar process folder : {base_path}")                                                       # 顯示實際處理資料夾
    process_subject_folder(base_path)                                                                            # 執行主流程
