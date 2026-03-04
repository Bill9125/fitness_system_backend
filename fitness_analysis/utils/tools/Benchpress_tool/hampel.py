import os
import pandas as pd
import numpy as np
import re

def hampel_filter(series, window_size=7, n_sigmas=3):
    """
    Hampel Filter for outlier detection.
    """
    series = np.array(series)
    n = len(series)
    k = 1.4826
    half_window = window_size // 2
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(half_window, n - half_window):
        window = series[i - half_window:i + half_window + 1]
        
        # Filter NaNs and check if enough points remain
        window_valid = window[~np.isnan(window)]
        if len(window_valid) < 3:
            continue

        median = np.median(window_valid)
        mad = k * np.median(np.abs(window_valid - median))
        if mad == 0:
            continue
        if np.abs(series[i] - median) > n_sigmas * mad:
            outlier_mask[i] = True

    return outlier_mask

def run_hampel_bar(folder_path):
    """
    Process bar coordinates with Hampel Filter.
    Equivalent to step0_hampel_bar.py
    """
    input_filename = "coordinates_interpolated.txt"
    input_path = os.path.join(folder_path, input_filename)

    if not os.path.exists(input_path):
        print(f"⛔ [Hampel Bar] 找不到檔案：{input_path}")
        return {}

    try:
        # 初始化數據存儲
        frames = []
        values = []

        # 讀取 YOLO 偵測數據
        with open(input_path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                frame_count = int(parts[0])  # 幀數
                frames.append(frame_count)
                values.append([float(parts[1]), float(parts[2])])
        
        values = np.array(values)
        x_outliers = hampel_filter(values[:, 0])
        y_outliers = hampel_filter(values[:, 1])

        outlier_frames = x_outliers | y_outliers
        values_filtered = values.copy().astype(float)
        values_filtered[outlier_frames] = np.nan

        results = {}
        for i, frame in enumerate(frames):
            results[frame] = [values_filtered[i][0], values_filtered[i][1]]
        
        print(f"✅ [Hampel Bar] 已處理完畢 (回傳資料量：{len(results)})")
        return results
    except Exception as e:
        print(f"❌ [Hampel Bar] 處理失敗：{input_path}, 錯誤：{e}")
        return {}

def process_skeleton_file(input_path, output_path, expected_joints):
    data = {} # frame -> {joint_idx: (x, y)}

    if not os.path.exists(input_path):
        return {}

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            frame_idx = int(parts[0])
            joint_idx = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            
            if frame_idx not in data:
                data[frame_idx] = {}
            data[frame_idx][joint_idx] = (x, y)

    all_frames = sorted(data.keys())
    all_points = []
    valid_frames = []
    
    for f_idx in all_frames:
        joints = data[f_idx]
        if len(joints) >= expected_joints:
            p_list = []
            for j_idx in range(expected_joints):
                if j_idx in joints:
                    p_list.append(joints[j_idx])
                else:
                    p_list.append((np.nan, np.nan))
            all_points.append(p_list)
            valid_frames.append(f_idx)

    if len(all_points) == 0:
        return {}

    all_points = np.array(all_points) # (num_frames, expected_joints, 2)
    mask = np.zeros_like(all_points[:, :, 0], dtype=bool)

    for joint_idx in range(expected_joints):
        x_series = all_points[:, joint_idx, 0]
        y_series = all_points[:, joint_idx, 1]
        x_outliers = hampel_filter(x_series)
        y_outliers = hampel_filter(y_series)
        mask[:, joint_idx] = x_outliers | y_outliers

    cleaned_points = all_points.copy().astype(float)
    cleaned_points[mask] = np.nan

    results = {}
    for i, frame in enumerate(valid_frames):
        coords_flat = []
        for j in range(expected_joints):
            coords_flat.extend([cleaned_points[i, j, 0], cleaned_points[i, j, 1]])
        results[frame] = coords_flat
    return results

def run_hampel_yolo_ske_rear(folder_path):
    """
    Process rear view skeleton with Hampel Filter.
    Equivalent to step0_hampel_yolo_ske_rear.py
    """
    input_path = os.path.join(folder_path, "interpolated_skeleton_rear.txt")
    
    data = process_skeleton_file(input_path, None, expected_joints=6)
    if data:
        print(f"✅ [Hampel Rear] 已處理完畢 (回傳資料量：{len(data)})")
    else:
        print(f"⚠️ [Hampel Rear] 無有效資料或找不到檔案：{input_path}")
    return data

def run_hampel_yolo_ske_top(folder_path):
    """
    Process top view skeleton with Hampel Filter.
    Equivalent to step0_hampel_yolo_ske_top.py
    """
    input_path = os.path.join(folder_path, "interpolated_skeleton_top.txt")
    
    data = process_skeleton_file(input_path, None, expected_joints=8)
    if data:
        print(f"✅ [Hampel Top] 已處理完畢 (回傳資料量：{len(data)})")
    else:
        print(f"⚠️ [Hampel Top] 無有效資料或找不到檔案：{input_path}")
    return data
