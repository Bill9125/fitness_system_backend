import os
import numpy as np
import re

def hampel_filter(series, window_size=7, n_sigmas=3):
    n = len(series)
    k = 1.4826
    half_window = window_size // 2
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(half_window, n - half_window):
        window = series[i - half_window:i + half_window + 1]
        median = np.nanmedian(window)
        mad = k * np.nanmedian(np.abs(window - median))
        if mad == 0:
            continue
        if np.abs(series[i] - median) > n_sigmas * mad:
            outlier_mask[i] = True

    return outlier_mask

def parse_skeleton_line(line):
    match = re.match(r"Frame (\d+): \[\[(.+)\]\]", line.strip())
    if not match:
        return None, None
    frame_idx = int(match.group(1))
    coord_str = match.group(2)

    # 抓所有像 (x, y) 的對
    coord_parts = re.findall(r"\(([^)]+)\)", coord_str)
    coords = []
    for part in coord_parts:
        try:
            x_str, y_str = part.strip().split(",")
            x, y = float(x_str), float(y_str)
            coords.append((x, y))
        except Exception as e:
            print(f"⚠️ 解析失敗：'{part}'，錯誤訊息：{e}")
            coords.append((np.nan, np.nan))  # 無法解析的點標為 NaN
    return frame_idx, coords

def process_skeleton_file(input_path, output_path):
    all_frames = []
    all_points = []

    with open(input_path, "r") as f:
        for line in f:
            frame_idx, coords = parse_skeleton_line(line)
            if coords is not None and len(coords) == 6:
                all_frames.append(frame_idx)
                all_points.append(coords)

    if len(all_points) == 0:
        print(f"⚠️ 無有效資料：{input_path}")
        return False

    all_points = np.array(all_points)  # (num_frames, 6, 2)
    mask = np.zeros_like(all_points[:, :, 0], dtype=bool)

    for joint_idx in range(6):
        x_series = all_points[:, joint_idx, 0]
        y_series = all_points[:, joint_idx, 1]
        x_outliers = hampel_filter(x_series)
        y_outliers = hampel_filter(y_series)
        mask[:, joint_idx] = x_outliers | y_outliers

    cleaned_points = all_points.copy()
    cleaned_points[mask] = np.nan

    with open(output_path, "w") as f:
        for i, frame in enumerate(all_frames):
            coords_str = ", ".join(
                f"(NaN, NaN)" if np.isnan(x) or np.isnan(y) else f"({x}, {y})"
                for x, y in cleaned_points[i]
            )
            f.write(f"Frame {frame}: [[{coords_str}]]\n")

    return True

def process_all_skeletons(root_dir):
    processed_files = []

    # ✅ 只處理 root_dir 這個資料夾，不含子資料夾
    files = os.listdir(root_dir)
    for file in files:
        if file == "yolo_skeleton.txt":
            input_path = os.path.join(root_dir, file)
            output_path = os.path.join(root_dir, "yolo_skeleton_hampel.txt")

            # 👉 若已存在就跳過
            if os.path.exists(output_path):
                print(f"⏭️ 已存在，跳過：{output_path}")
                continue

            success = process_skeleton_file(input_path, output_path)
            if success:
                print(f"✅ 已處理：{output_path}")
                processed_files.append(output_path)

    # === 處理報告 ===
    print("\n📋 處理完成列表：")
    for path in processed_files:
        print(f"  ➤ {path}")
    print(f"\n✅ 共處理 {len(processed_files)} 個檔案")





if __name__ == "__main__":                                                                                               # 入口
    import sys                                                                                                           # 取參數
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                             # 預設根目錄
    # 用法：python this_script.py <folder_path>                                                                           # 說明

    if len(sys.argv) >= 2:                                                                                               # 有傳入路徑
        base_path = sys.argv[1]                                                                                          # 取第一參數
        if not os.path.isdir(base_path):                                                                                 # 確認是資料夾
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                                # 拋錯
    else:                                                                                                                # 無參數 → 退回找最新
        if not os.path.isdir(recordings_dir):                                                                            # 根目錄要存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                                      # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)                                      # 列子資料夾
                if os.path.isdir(os.path.join(recordings_dir, d))]                                                       # 只要資料夾
        if not subs:                                                                                                     # 無子資料夾
            raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾，且未提供參數")                                # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                                      # 取最新一個

    print(f"▶ 處理資料夾：{base_path}")                                                                                   # 顯示實際處理對象
    process_all_skeletons(base_path)                                                                                     # 執行主流程

