import os
import numpy as np

def linear_interpolate(data, target_length=100):
    current_length = len(data)
    if current_length == target_length:
        return data
    x_original = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_original, data)

def process_cut_folder(folder_path, data_root_path):
    subject_id = os.path.basename(os.path.dirname(folder_path))
    cut_number = os.path.basename(folder_path).replace("cut4_", "")
    new_folder = f"cut4_{cut_number}_length100_data"
    new_folder_path = os.path.join(folder_path, new_folder)
    print(f"📌 DEBUG 受試者: {subject_id}, cut: {cut_number}")

    file_map = {
        "right_torso_arm_angle_11m.txt": "right_torso_arm_angle_11m_100.txt",
        "left_torso_arm_angle_11m.txt": "left_torso_arm_angle_11m_100.txt",
        "right_wrist_to_shoulderline_11m.txt": "right_wrist_to_shoulderline_11m_100.txt",
        "left_wrist_to_shoulderline_11m.txt": "left_wrist_to_shoulderline_11m_100.txt",
        "bar_x.txt": "bar_100x.txt",
        "bar_y.txt": "bar_100y.txt",
        "bar_y_div_x.txt": "bar_y_div_x_100.txt",
        "left_elbow_angle.txt": "left_elbow_angle_100.txt",
        "left_shoulder_angle.txt": "left_shoulder_angle_100.txt",
        "right_elbow_angle.txt": "right_elbow_angle_100.txt",
        "right_shoulder_angle.txt": "right_shoulder_angle_100.txt",
        "left_shoulder_y.txt": "left_shoulder_100y.txt",
        "right_shoulder_y.txt": "right_shoulder_100y.txt"
    }

    if os.path.exists(new_folder_path):
        all_exist = all(os.path.exists(os.path.join(new_folder_path, out_file)) for out_file in file_map.values())
        if all_exist:
            print(f"⏭️ 已內插所有特徵，跳過: {new_folder_path}")
            return
    else:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"📂 建立資料夾: {new_folder_path}")

    for in_file, out_file in file_map.items():
        in_path = os.path.join(data_root_path, in_file)  # ✅ 改成從 root 抓資料
        if os.path.exists(in_path):
            data = np.loadtxt(in_path, delimiter=',')
            if data.ndim > 1:
                data = data[:, 1]
            interp = linear_interpolate(data)
            np.savetxt(os.path.join(new_folder_path, out_file), interp, fmt="%.6f")
            print(f"✅ 內插完成: {in_file} → {out_file}")
        else:
            print(f"⚠️ 缺少檔案: {in_file}")


if __name__ == "__main__":                                                                                  # 入口
    import sys, os                                                                                           # 匯入
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                 # 預設 recordings 根目錄

    if len(sys.argv) >= 2:                                                                                   # 有傳入資料夾參數（預期是 recording_* 目錄）
        data_root = sys.argv[1]                                                                              # .txt 真實來源於 recording_* 根目錄
        if not os.path.isdir(data_root):                                                                     # 檢查有效性
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{data_root}")                                      # 拋錯
    else:                                                                                                    # 沒傳參數 → 退回最新
        if not os.path.isdir(recordings_dir):                                                                # 根目錄存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                          # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)
                if os.path.isdir(os.path.join(recordings_dir, d))]                                           # 列子資料夾
        if not subs:                                                                                         # 無子資料夾
            raise FileNotFoundError("❌ recordings 下沒有任何子資料夾，且未提供參數")                          # 拋錯
        data_root = max(subs, key=os.path.getmtime)                                                          # 取最新 recording_* 作為 data_root

    base_path = os.path.join(data_root, "feature")                                                           # 切割輸出所在的 feature 目錄
    if not os.path.isdir(base_path):                                                                         # 檢查 feature 目錄
        raise FileNotFoundError(f"❌ 找不到 feature 目錄：{base_path}")                                       # 拋錯

    cut_folders = [os.path.join(base_path, f) for f in os.listdir(base_path)
                   if f.startswith("cut4_") and os.path.isdir(os.path.join(base_path, f))]                   # 收集 cut4_* 目錄
    if not cut_folders:                                                                                      # 空集合
        raise FileNotFoundError("❌ 找不到任何 cut4 資料夾")                                                  # 拋錯

    for cut_folder in cut_folders:                                                                           # 逐一處理
        print(f"\n🔍 處理資料夾: {cut_folder}")                                                               # 顯示進度
        process_cut_folder(cut_folder, data_root)                                                            # 帶入 data_root 執行

    print("✅ Done")                                                                                         # 完成提示

