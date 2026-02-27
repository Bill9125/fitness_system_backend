# -*- coding: utf-8 -*-                                                                                     # 檔頭宣告
import os, sys, re, json, math, ast                                                                         # 基本匯入
import numpy as np                                                                                           # 向量/角度計算

# === 既定骨架檔名（固定） ===
SKE_TXT_NAME = "yolo_skeleton_top_11m_interpolated_hampel.txt"                                               # 骨架txt檔名

# === 點位索引：0左肩,1右肩,2左髖,3右髖,4左肘,5右肘,6左腕,7右腕 ===
L_SHO, R_SHO, L_HIP, R_HIP, L_ELB, R_ELB = 0, 1, 2, 3, 4, 5                                                  # 索引簡寫

def angle_abc(a, b, c):                                                                                      # 計算∠ABC（頂點B）
    a = np.asarray(a, dtype=float)                                                                            # 轉float陣列
    b = np.asarray(b, dtype=float)                                                                            # 轉float陣列
    c = np.asarray(c, dtype=float)                                                                            # 轉float陣列
    v1, v2 = a - b, c - b                                                                                     # 邊向量
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)                                                            # 邊長
    if n1 == 0 or n2 == 0:                                                                                    # 邊長為0保護
        return None                                                                                           # 回None
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)                                                     # 夾角cos且夾範圍
    return float(np.degrees(np.arccos(cosv)))                                                                 # 轉度數回傳

def parse_skeleton_txt(txt_path):                                                                             # 解析骨架txt
    frames, kps_all = [], []                                                                                  # 幀序與每幀關鍵點
    pat = re.compile(r"^Frame\s+(\d+):\s*(\[.*\])\s*$")                                                       # 行樣式
    with open(txt_path, "r", encoding="utf-8") as f:                                                          # 開檔
        for line in f:                                                                                        # 逐行
            m = pat.match(line.strip())                                                                       # 比對
            if not m:                                                                                         # 不符合
                continue                                                                                      # 跳過
            fid = int(m.group(1))                                                                             # 幀號
            raw = m.group(2)                                                                                  # 資料字串
            try:
                data = ast.literal_eval(raw)                                                                  # 安全解析
            except Exception:
                data = []                                                                                     # 解析失敗給空
            kp_list = data[0] if isinstance(data, list) and len(data) > 0 else []                             # 取第一人
            frames.append(fid)                                                                                # 存幀號
            kps_all.append(kp_list)                                                                           # 存關鍵點
    # 依幀號排序，確保 frames/points 對齊（若檔案未排序）                                                             # 排序保護
    order = np.argsort(frames)                                                                                # 排序索引
    frames = [frames[i] for i in order]                                                                       # 重排幀
    kps_all = [kps_all[i] for i in order]                                                                     # 重排點
    return frames, kps_all                                                                                    # 回傳

def compute_angles(frames, kps_all):                                                                          # 計算左右角度
    left_vals, right_vals = [], []                                                                            # 左右序列
    for kp in kps_all:                                                                                        # 逐幀
        if not kp or len(kp) < 6:                                                                             # 點不足
            left_vals.append(None); right_vals.append(None)                                                   # 填None
            continue                                                                                          # 下一幀
        try:
            angL = angle_abc(kp[L_ELB], kp[L_SHO], kp[L_HIP])                                                 # 左：4-0-2
            angR = angle_abc(kp[R_ELB], kp[R_SHO], kp[R_HIP])                                                 # 右：5-1-3
        except Exception:
            angL, angR = None, None                                                                           # 任何索引錯誤
        left_vals.append(None if (angL is None or (isinstance(angL, float) and math.isnan(angL))) else angL)  # 收左
        right_vals.append(None if (angR is None or (isinstance(angR, float) and math.isnan(angR))) else angR) # 收右
    return left_vals, right_vals                                                                              # 回傳

def dump_angle_json(out_path, title, frames, values):                                                         # 輸出JSON
    reals = [v for v in values if v is not None]                                                              # 實值列表
    y_min = float(np.min(reals)) if reals else None                                                           # y最小
    y_max = float(np.max(reals)) if reals else None                                                           # y最大
    payload = {                                                                                               # 組物件
        "title": title,                                                                                       # 標題
        "y_label": "Angle (degrees)",                                                                         # Y標籤
        "y_min": y_min,                                                                                       # y_min
        "y_max": y_max,                                                                                       # y_max
        "frames": frames,                                                                                     # 幀序列
        "values": values                                                                                      # 角度序列
    }
    with open(out_path, "w", encoding="utf-8") as f:                                                          # 開檔
        json.dump(payload, f, ensure_ascii=False, indent=4)                                                   # 寫檔

def main(folder_path):                                                                                        # 主流程
    if not os.path.isdir(folder_path):                                                                        # 資料夾檢查
        raise NotADirectoryError(f"不是有效資料夾：{folder_path}")                                              # 拋錯
    txt_path = os.path.join(folder_path, SKE_TXT_NAME)                                                        # 組骨架路徑
    if not os.path.exists(txt_path):                                                                          # 檔案檢查
        raise FileNotFoundError(f"找不到骨架txt：{txt_path}")                                                   # 拋錯

    frames, kps_all = parse_skeleton_txt(txt_path)                                                            # 讀取解析
    if not frames:                                                                                            # 無資料
        raise ValueError("骨架txt沒有可解析的 Frame 資料")                                                       # 拋錯

    left_vals, right_vals = compute_angles(frames, kps_all)                                                   # 計算角度

    out_left  = os.path.join(folder_path, "left_elbow_torsor_angle_top.json")                                  # 左JSON路徑
    out_right = os.path.join(folder_path, "right_elbow_torsor_angle_top.json")                                 # 右JSON路徑
    dump_angle_json(out_left,  "Left Elbow–Trunk Angle (4-0-2)",  frames, left_vals)                          # 寫左JSON
    dump_angle_json(out_right, "Right Elbow–Trunk Angle (5-1-3)", frames, right_vals)                         # 寫右JSON

    print("✅ 完成！")                                                                                          # 提示
    print(f"➡️ 左手角度 JSON：{out_left}")                                                                     # 顯示路徑
    print(f"➡️ 右手角度 JSON：{out_right}")                                                                    # 顯示路徑

# if __name__ == "__main__":                                                                                    # 進入點
#     folder = sys.argv[1] if len(sys.argv) > 1 else "."                                                        # 取資料夾參數
#     main(folder)                                                                                               # 執行

if __name__ == "__main__":                                                                                   # 進入點
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                 # 預設 recordings 根目錄
    if len(sys.argv) >= 2:                                                                                   # 有傳入資料夾參數
        base_path = sys.argv[1]                                                                              # 取第一個參數
        if not os.path.isdir(base_path):                                                                     # 檢查有效性
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                     # 拋錯
    else:                                                                                                     # 無參數 → 退回找最新
        if not os.path.isdir(recordings_dir):                                                                 # recordings 根目錄要存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                           # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)
                if os.path.isdir(os.path.join(recordings_dir, d))]                                           # 列出所有子資料夾
        if not subs:                                                                                          # 無任何子資料夾
            raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾，且未提供參數")                      # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                           # 取最新子資料夾
    print(f"▶ 處理資料夾：{base_path}")                                                                        # 顯示此次實際處理路徑
    main(base_path)                                                                                           # 執行主流程
