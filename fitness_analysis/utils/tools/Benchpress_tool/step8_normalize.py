import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# ====== 資料清洗與正規化函數 ======
def remove_outliers_and_interpolate(data):
    if len(data) < 3:
        return data
    mean = np.mean(data)
    std_dev = np.std(data)
    lower, upper = mean - 3 * std_dev, mean + 3 * std_dev
    data_cleaned = np.copy(data)
    outliers = (data < lower) | (data > upper)
    data_cleaned[outliers] = np.nan
    indices = np.arange(len(data_cleaned))
    valid = ~np.isnan(data_cleaned)
    if np.sum(valid) > 1:
        return np.interp(indices, indices[valid], data_cleaned[valid])
    else:
        return np.full_like(data, mean)

def variation_normalize(data):
    out = np.zeros(len(data))
    for i in range(1, len(data)):
        out[i] = data[i-1] - data[i]
    return out

def variation_acceleration_normalize(data):
    out = np.zeros(len(data))
    for i in range(2, len(data)):
        out[i] = (data[i] - data[i-1]) - (data[i-1] - data[i-2])
    return out

def variation_ratio_normalize(data):
    out = np.zeros(len(data))
    for i in range(1, len(data)):
        if data[i-1] != 0:
            out[i] = (data[i-1] - data[i]) / data[i-1]
        else:
            out[i] = 0
    return out

def z_score_normalize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def min_max_normalize(data):
    min_val, max_val = np.min(data), np.max(data)
    return 2 * (data - min_val) / (max_val - min_val + 1e-8) - 1


# ====== 單一 cut4_n_length100_data 的處理函式 ======
def process_single_data_folder(data_folder_path):
    print(f"\n📁 處理: {data_folder_path}")

    # Step 1: Remove outliers
    outlier_path = os.path.join(data_folder_path, "remove_outlier")
    os.makedirs(outlier_path, exist_ok=True)

    for txt_file in os.listdir(data_folder_path):
        if txt_file.endswith(".txt"):
            src = os.path.join(data_folder_path, txt_file)
            dst = os.path.join(outlier_path, txt_file)

            if os.path.exists(dst):
                print(f"⏭️ 已存在，略過: {txt_file}")
                continue

            try:
                raw_data = np.loadtxt(src)
                if raw_data.ndim > 1:
                    raw_data = raw_data[:, 0]
                clean_data = remove_outliers_and_interpolate(raw_data)
                np.savetxt(dst, clean_data, fmt="%.6f")
                print(f"✅ outlier removed: {txt_file}")
            except Exception as e:
                print(f"❌ outlier error {txt_file}: {e}")

    # Step 2: Normalize
    norm_path = os.path.join(data_folder_path, "normalize_remove_outliner")
    os.makedirs(norm_path, exist_ok=True)
    files_to_normalize = [f for f in os.listdir(outlier_path) if f.endswith(".txt")]

    for txt_file in files_to_normalize:
        try:
            base = txt_file.replace(".txt", "")
            v1_path = os.path.join(norm_path, f"{base}_Variation.txt")
            v2_path = os.path.join(norm_path, f"{base}_Variation_acceleration.txt")
            vr_path = os.path.join(norm_path, f"{base}_Variation_ratio.txt")
            z_path = os.path.join(norm_path, f"{base}_z_score.txt")

            if all(os.path.exists(p) for p in [v1_path, v2_path, vr_path, z_path]):
                print(f"⏭️ 已存在，略過: {txt_file}")
                continue

            data = np.loadtxt(os.path.join(outlier_path, txt_file))
            if data.ndim > 1:
                data = data[:, 0]
            v1 = variation_normalize(data)
            v2 = variation_acceleration_normalize(data)
            vr = variation_ratio_normalize(data)
            z = z_score_normalize(data)
            np.savetxt(v1_path, v1, fmt="%.6f")
            np.savetxt(v2_path, v2, fmt="%.6f")
            np.savetxt(vr_path, vr, fmt="%.6f")
            np.savetxt(z_path, z, fmt="%.6f")
            print(f"✅ normalized: {txt_file}")
        except Exception as e:
            print(f"❌ normalize error {txt_file}: {e}")

    # Step 3: Min-max normalization
    minmax_path = os.path.join(data_folder_path, "min_max_normalize")
    os.makedirs(minmax_path, exist_ok=True)
    for txt_file in os.listdir(norm_path):
        if txt_file.endswith(".txt"):
            try:
                src_path = os.path.join(norm_path, txt_file)
                out_name = txt_file.replace(".txt", "_min_max.txt")
                out_path = os.path.join(minmax_path, out_name)
                if os.path.exists(out_path):
                    print(f"⏭️ 已存在，略過: {txt_file}")
                    continue
                data = np.loadtxt(src_path)
                normalized = min_max_normalize(data)
                np.savetxt(out_path, normalized, fmt="%.6f")
                print(f"✅ min-max done: {txt_file}")
            except Exception as e:
                print(f"❌ min-max error {txt_file}: {e}")


# ====== 改寫新版的全資料處理流程 ======
def process_all_length100_folders(base_path):
    feature_path = os.path.join(base_path, "feature")
    if not os.path.exists(feature_path):
        raise FileNotFoundError("❌ 找不到 feature 資料夾")

    for folder in os.listdir(feature_path):
        cut_path = os.path.join(feature_path, folder)
        if not (os.path.isdir(cut_path) and folder.startswith("cut4_")):
            continue

        for subfolder in os.listdir(cut_path):
            if subfolder.endswith("_length100_data"):
                data_folder_path = os.path.join(cut_path, subfolder)
                process_single_data_folder(data_folder_path)


# ====== 程式入口點 ======
if __name__ == "__main__":                                                                                  # 入口
    import sys, os                                                                                           # 匯入
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                 # 預設 recordings 根目錄

    if len(sys.argv) >= 2:                                                                                   # 有傳入資料夾參數
        base_path = sys.argv[1]                                                                              # 指定 recording_* 根目錄
        if not os.path.isdir(base_path):                                                                     # 檢查有效性
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                      # 拋錯
    else:                                                                                                    # 沒傳參數 → 退回最新
        if not os.path.isdir(recordings_dir):                                                                # 根目錄存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                          # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)
                if os.path.isdir(os.path.join(recordings_dir, d))]                                           # 列子資料夾
        if not subs:                                                                                         # 無子資料夾
            raise FileNotFoundError("❌ recordings 下沒有任何子資料夾，且未提供參數")                          # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                          # 取最新 recording_* 目錄

    print(f"📌 使用資料夾: {base_path}")                                                                     # 顯示本次處理對象
    process_all_length100_folders(base_path)                                                                 # 執行主流程
    print("✅ 全部處理完成")                                                                                  # 完成提示

