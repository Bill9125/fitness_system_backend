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
            if coords is not None and len(coords) == 8:
                all_frames.append(frame_idx)
                all_points.append(coords)

    if len(all_points) == 0:
        print(f"⚠️ 無有效資料：{input_path}")
        return False

    all_points = np.array(all_points)  # (num_frames, 8, 2)
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
    files = os.listdir(root_dir)  # 不用 walk，只取該資料夾
    for file in files:
        if file == "yolo_skeleton_top.txt":
            input_path = os.path.join(root_dir, file)
            output_path = os.path.join(root_dir, "yolo_skeleton_top_11m_hampel.txt")
            success = process_skeleton_file(input_path, output_path)
            if success:
                print(f"✅ 已處理：{output_path}")
                processed_files.append(output_path)

    # === 處理報告 ===
    print("\n📋 處理完成列表：")
    for path in processed_files:
        print(f"  ➤ {path}")
    print(f"\n✅ 共處理 {len(processed_files)} 個檔案")


if __name__ == "__main__":                                                                                      # 程式入口
    import sys, os                                                                                              # 匯入模組
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                    # 預設 recordings 路徑

    # === 優先順序：CLI 傳入路徑 > 自動取最新錄影資料夾 ===
    if len(sys.argv) >= 2:                                                                                      # 若有傳入資料夾參數
        base_path = sys.argv[1]                                                                                 # 取第一個參數
        if not os.path.isdir(base_path):                                                                        # 防呆：檢查是否為資料夾
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                       # 錯誤提示
    else:                                                                                                       # 若未傳參數則取 recordings 下最新
        if not os.path.isdir(recordings_dir):                                                                   # recordings 必須存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                             # 拋錯
        subfolders = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)
                      if os.path.isdir(os.path.join(recordings_dir, d))]                                        # 取得所有子資料夾
        if not subfolders:                                                                                      # 若 recordings 下沒資料夾
            raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾，且未提供參數")                       # 拋錯
        base_path = max(subfolders, key=os.path.getmtime)                                                       # 取最新錄影資料夾

    print(f"🏋️ hamp ske top process folder : {base_path}")                                                      # 顯示實際處理資料夾
    process_all_skeletons(base_path)                                                                            # 執行主函式
   