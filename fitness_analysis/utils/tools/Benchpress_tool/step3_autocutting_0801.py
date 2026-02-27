import numpy as np  # 引入數值運算套件
import ast  # 匯入 abstract syntax trees 用來轉換文字為資料結構
import re  # 匯入正規表達式模組
import pandas as pd  # 匯入 pandas 用來操作表格資料
from scipy.signal import find_peaks  # 匯入尋找極值點的函數
import matplotlib.pyplot as plt  # 匯入 matplotlib 畫圖模組
import xlsxwriter  # 匯入寫 Excel 的模組
import os  # 匯入作業系統操作模組
from openpyxl import load_workbook
from scipy.ndimage import uniform_filter1d
import math


#骨架資料運算
def get_angle(a, b, c):  # 計算 ∠abc 的角度（單位為度）
    ab = np.array(a) - np.array(b)  # 向量 ab
    cb = np.array(c) - np.array(b)  # 向量 cb
    dot = np.dot(ab, cb)  # ab 與 cb 的內積
    norm_ab = np.linalg.norm(ab)  # ab 的長度
    norm_cb = np.linalg.norm(cb)  # cb 的長度
    if norm_ab == 0 or norm_cb == 0:  # 若向量長度為 0 則無法計算
        return None  # 回傳 None 表示無效角度
    angle = np.arccos(np.clip(dot / (norm_ab * norm_cb), -1.0, 1.0))  # 夾角（弧度）
    angle_deg = np.degrees(angle)  # 轉為角度
    return angle_deg if angle_deg <= 180 else 360 - angle_deg  # 若大於 180 則反轉

def parse_frame_data(input_file):  # 解析每一幀的資料並計算關節角度
    results = []  # 初始化儲存結果的 list
    with open(input_file, "r", encoding="utf-8") as f:  # 開啟檔案
        for idx, line in enumerate(f):  # 一行一行處理
            try:
                cleaned_line = re.sub(r"^Frame\s*\d+:\s*", "", line.strip())  # 去除前綴
                frame_data = ast.literal_eval(cleaned_line)[0]  # 將文字轉為 list
                if len(frame_data) < 6:  # 不足六個點跳過
                    continue  # 跳到下一行
                left_shoulder = frame_data[0]  # 左肩座標
                right_shoulder = frame_data[1]  # 右肩座標
                left_elbow = frame_data[2]  # 左肘座標
                right_elbow = frame_data[3]  # 右肘座標
                left_wrist = frame_data[4]  # 左腕座標
                right_wrist = frame_data[5]  # 右腕座標

                left_elbow_angle = get_angle(left_wrist, left_elbow, left_shoulder)  # 左肘角
                right_elbow_angle = get_angle(right_wrist, right_elbow, right_shoulder)  # 右肘角
                left_shoulder_angle = get_angle(left_elbow, left_shoulder, right_shoulder)  # 左肩角
                right_shoulder_angle = get_angle(right_elbow, right_shoulder, left_shoulder)  # 右肩角
                avg_elbow_angle = np.mean([left_elbow_angle, right_elbow_angle])

                results.append([  # 儲存結果
                    idx + 1,  # 幀數
                    left_elbow_angle,
                    right_elbow_angle,
                    left_shoulder_angle,
                    right_shoulder_angle,
                    avg_elbow_angle
                ])
            except Exception as e:  # 發生錯誤
                print(f"錯誤發生在第 {idx+1} 幀: {e}")  # 顯示錯誤訊息
                continue  # 跳過該行
    return pd.DataFrame(results, columns=[  # 回傳為 DataFrame
        "Frame", "Left Elbow Angle", "Right Elbow Angle",  # 欄位名稱
        "Left Shoulder Angle", "Right Shoulder Angle", "avg_elbow_angle"
    ])


def filter_close_valleys(angle, valley_indices, peak_indices, min_peak_diff=60):
    """
    合併中間波峰不明顯的相鄰波谷，只保留較深的那一個。
    條件：
    - 兩個波谷中間沒有任何一個波峰，比較深波谷高出 min_peak_diff → 合併
    """
    filtered_valleys = []
    i = 0
    while i < len(valley_indices):
        current_valley = valley_indices[i]
        if i + 1 < len(valley_indices):
            next_valley = valley_indices[i + 1]
            middle_peaks = [p for p in peak_indices if current_valley < p < next_valley]

            deeper_valley = current_valley if angle[current_valley] < angle[next_valley] else next_valley

            # ➤ 條件判斷：若中間沒有任一波峰高出 min_peak_diff，視為假雙谷
            is_all_peaks_shallow = True
            for p in middle_peaks:
                if angle[p] - angle[deeper_valley] >= min_peak_diff:
                    is_all_peaks_shallow = False
                    break

            if is_all_peaks_shallow:
                filtered_valleys.append(deeper_valley)
                i += 2
                continue  # 跳過下一個 valley

        filtered_valleys.append(current_valley)
        i += 1
    return filtered_valleys


def find_clear_difference_side(angle, idx, min_peak_diff, max_search=70):
    val = angle[idx]  # 當前波谷值

    # 向左搜尋是否有明顯比它高的點
    left_idx = idx - 1
    while left_idx >= max(0, idx - max_search):
        if angle[left_idx] - val >= min_peak_diff:  # 僅接受高於波谷
            break
        left_idx -= 1
    else:
        return False  # 沒找到左側符合條件

    # 向右搜尋是否有明顯比它高的點
    right_idx = idx + 1
    while right_idx < min(len(angle), idx + max_search):
        if angle[right_idx] - val >= min_peak_diff:  # 僅接受高於波谷
            break
        right_idx += 1
    else:
        return False  # 沒找到右側符合條件

    return True  # 左右兩側都找到明顯高點


def is_clear_valley(angle, idx, window=25, min_prominence=0.05, future_window=18, slope_check=True, min_peak_diff=100, is_first=False):
    idx = int(idx)
    val = angle[idx]  # 取出當前波谷值

    start = max(0, idx - window)
    end = min(len(angle), idx + window)
    left = angle[start:idx]  # 左側 window 區段
    right = angle[idx+1:end]  # 右側 window 區段

    if len(left) == 0 or len(right) == 0:
        print(f"[{idx}] ⚠️ 無法取得足夠左右資料")
        return False

    left_max = np.max(left)  # 左側最大值
    right_max = np.max(right)  # 右側最大值

    # ✅ 條件1：相對 Prominence，波谷需顯著低於左右兩側高點
    dynamic_prominence = min(min_prominence * val, 20)  # 計算相對 prominence（限制最大值）
    condition_valley_shape = (
        val < left_max and
        val < right_max and
        (left_max - val > dynamic_prominence) or #0525改配合第三段過濾
        (right_max - val > dynamic_prominence)
    )
    if not condition_valley_shape:
        print(f"[{idx}] ❌ shape 不成立: val={val:.2f}, left_max={left_max:.2f}, right_max={right_max:.2f}, min_prom={dynamic_prominence:.2f}")
        # print(
        #     f"[{idx}] ❌ shape 不成立\n"
        #     f"    val         = {val:.2f}\n"
        #     f"    left_max    = {left_max:.2f}\n"
        #     f"    right_max   = {right_max:.2f}\n"
        #     f"    left_diff   = {left_max - val:.2f} vs required > {dynamic_prominence:.2f}\n"
        #     f"    right_diff  = {right_max - val:.2f} vs required > {dynamic_prominence:.2f}"
        # )
    # ✅ 條件2：未來30幀內不能比當前波谷更低
    future = angle[idx+1 : idx+1+future_window]
    condition_no_later_drop = all(val <= f for f in future)
    if not condition_no_later_drop:
        print(f"[{idx}] ❌ 未來有下降: val={val:.2f}, future={future[:5]}...")

    condition_peak_diff = find_clear_difference_side(angle, idx, min_peak_diff)


    # ✅ 條件6：波谷前段平均斜率需為下降
    condition_pre_slope = True
    past_window = 20      #0709原本是15
    if slope_check and idx - past_window >= 0:
        prev = angle[idx - past_window:idx]
        pre_slope = np.diff(prev)
        mean_pre_slope = np.mean(pre_slope)
        condition_pre_slope = mean_pre_slope < 0
        if not condition_pre_slope:
            print(f"[{idx}] ❌ 前段斜率不符: mean_pre_slope={mean_pre_slope:.4f}")

    # ✅ 條件7：波谷後段平均斜率需為上升
    condition_post_slope_tail = True
    if slope_check and idx + past_window < len(angle):
        post = angle[idx + 1:idx + 1 + past_window]  # 從 idx+1 開始取
        post_slope = np.diff(post)
        mean_post_slope = np.mean(post_slope)
        condition_post_slope_tail = mean_post_slope > 0
        if not condition_post_slope_tail:
            print(f"[{idx}] ❌ 後段斜率不符: mean_post_slope={mean_post_slope:.4f}")


    return (
        condition_valley_shape and
        condition_no_later_drop and
        # condition_slope and
        condition_peak_diff and
        #condition_left_check and
        condition_pre_slope and
        condition_post_slope_tail
    )

def find_corners_by_curvature_with_start_end(kappa, angle, valley, peaks, threshold=20, verbose=True):
    """
    用於以 peak-valley-peak 為中心，切出左右側的曲率最低點作為 start/end。
    - 若曲率不符條件，則 fallback 使用原始 peak。
    - 自動補上左右端點的假 peak，避免 IndexError。
    
    回傳：
    - ends: 每段結束點（右側）
    - starts: 每段開始點（左側）
    """

    starts = []
    ends = []

    # === 主迴圈：依 valley 找對應左右段落 ===
    for i in range(len(valley)):
        left_peak = peaks[i]
        right_peak = peaks[i + 1]
        v = valley[i]

        # --- start：在 left_peak ~ valley 之間找最凹的點 ---
        seg_start = np.arange(left_peak, v)
        valid_start = None
        if len(seg_start) > 0:
            kappa_seg = kappa[seg_start]
            sorted_idx = np.argsort(kappa_seg)  # 負最大在前
            for idx in sorted_idx:
                p_idx = seg_start[idx]
                if angle[p_idx] - angle[v] >= threshold:
                    valid_start = p_idx
                    break
        if valid_start is None:
            valid_start = left_peak
        starts.append(valid_start)

        # --- end：在 valley ~ right_peak 之間找最凹的點（反向）---
        seg_end = np.arange(right_peak, v, -1)
        valid_end = None
        if len(seg_end) > 0:
            kappa_seg = kappa[seg_end]
            sorted_idx = np.argsort(kappa_seg)
            for idx in sorted_idx:
                p_idx = seg_end[idx]
                if angle[p_idx] - angle[v] >= threshold:
                    valid_end = p_idx
                    break
        if valid_end is None:
            valid_end = right_peak
        ends.append(valid_end)

        # # --- Debug 顯示 ---
        # if verbose:
        #     print(f"[Segment {i+1}] peakL={left_peak}, valley={v}, peakR={right_peak} | start={valid_start}, end={valid_end}")

    return ends, starts





def analyze_valleys_peaks(df, min_peak_diff=60):  # ✅ threshold_ratio 已移除
    angle = df["smoothing_uniform_1"].astype(float).values  # 使用平滑後的 bar_y_axis
    inverted_angle = -angle  # 用於找波谷

    # 計算 bar_position：最後5個非NaN值的平均
    valid_tail = df["smoothing_uniform_1"].dropna().values[-5:]
    bar_position = np.mean(valid_tail) +20
    print(f"📏 bar_position 門檻值: {bar_position:.2f}")

    mean = np.nanmean(angle)
    std = np.nanstd(angle)
    filtered = angle[(angle > mean - 2 * std) & (angle < mean + 2 * std)]
    if len(filtered) == 0:
        raise ValueError("角度資料全部為異常值，無法計算門檻")
    
    max_val = np.max(filtered)
    min_val = np.min(filtered)
    range_val = max_val - min_val
    print(f"max_val is {max_val}, min_val is {min_val}, range_val is {range_val}")

    #peak_indices, _ = find_peaks(angle, distance=25, prominence=0.1, width=2)
    valley_indices_raw, _ = find_peaks(inverted_angle, distance=25, prominence=0.1, width=2)

    print(f"原始 valley_indices_all: {valley_indices_raw}")

# --- 第一階段波谷過濾（含 bar_position 條件） ---
    valley_indices = []
    first_valid_idx = None
    for idx in valley_indices_raw:
        is_first = first_valid_idx is None
        is_valid = is_clear_valley(angle, idx, window=25, min_peak_diff=min_peak_diff, is_first=is_first)
        if is_valid and angle[idx] < bar_position:  # ✅ 加入 bar_position 條件
            if first_valid_idx is None:
                first_valid_idx = idx
            valley_indices.append(int(idx))
    print(f"✅ 第一階段 valley_indices（含 bar_position 過濾）: {valley_indices}")

    # --- 原始波峰 ---
    peak_indices_raw, _ = find_peaks(angle, distance=25, prominence=0.1, width=2)

    # --- 波峰過濾 ---
    peak_indices = []
    for p_idx in peak_indices_raw:
        if len(valley_indices) == 0:
            continue
        nearest_valley = min(valley_indices, key=lambda v: abs(v - p_idx))
        peak_val = angle[p_idx]
        valley_val = angle[nearest_valley]
        diff = peak_val - valley_val
        if diff >= min_peak_diff:
            peak_indices.append(p_idx)
        else:
            print(f"[{p_idx}] ❌ 假波峰，與 valley[{nearest_valley}] 差值 {diff:.2f} < {min_peak_diff}")
    print(f"原始 peak_indices_raw: {peak_indices_raw}")    
    print(f"過濾後的 peak_indices: {peak_indices}")    

    # --- 假雙谷合併 ---
    valley_indices = filter_close_valleys(angle, valley_indices, peak_indices, min_peak_diff=min_peak_diff)
    print(f"假雙谷合併後 valley_indices: {valley_indices}")

    # --- 第二階段過濾 ---
    print(f"-------------第二次過濾開始-------------")
    final_valleys = []
    for i, v_idx in enumerate(valley_indices):
        left_peaks = [p for p in peak_indices if p < v_idx]
        right_peaks = [p for p in peak_indices if p > v_idx]

        if i == 0:
            if not right_peaks:
                print(f"[{v_idx}] ❌ 第一個波谷缺右側波峰")
                continue
            right_peak = min(right_peaks)
            right_diff = angle[right_peak] - angle[v_idx]
            if right_diff >= min_peak_diff:
                final_valleys.append(v_idx)
            else:
                print(f"[{v_idx}] ❌ 第一個波谷右差值不夠: {right_diff:.2f}")
            continue

        if not left_peaks or not right_peaks:
            print(f"[{v_idx}] ❌ 缺少對應波峰：左側={left_peaks}, 右側={right_peaks}")
            continue

        left_peak = max(left_peaks)
        right_peak = min(right_peaks)
        left_diff = angle[left_peak] - angle[v_idx]
        right_diff = angle[right_peak] - angle[v_idx]

        if left_diff >= min_peak_diff or right_diff >= min_peak_diff:
            final_valleys.append(v_idx)
        else:
            print(f"[{v_idx}] ❌ 差值不夠：left_diff={left_diff:.2f}, right_diff={right_diff:.2f}")

    # --- 更新 valley_indices（第二階段通過者） ---
    valley_indices = final_valleys
    print(f"✅ 最終 valley_indices（兩階段過濾）: {valley_indices}")

    # --- 最終回傳結果 ---
    return angle, peak_indices, valley_indices


def compute_derivatives(df, side):
    """
    根據 side='left' 或 'right'，計算對應手肘的 dy, ddy, kappa 並寫入欄位。
    只需輸入 'left' 或 'right'。
    """
    if side not in ["left", "right"]:
        raise ValueError("❌ side 必須為 'left' 或 'right'")

    col_name = f"{side}_elbow_angle_smoothing"  # 自動組合欄位名稱
    if col_name not in df.columns:
        raise ValueError(f"❌ 欄位 '{col_name}' 不存在於 df 中")

    angle = df[col_name].values  # 取角度欄位
    dy = np.gradient(angle)  # 一階導數
    ddy = np.gradient(dy)  # 二階導數
    kappa = ddy / np.power(1 + dy**2, 1.5)  # 曲率公式

    df[f"dy_{side}"] = dy
    df[f"ddy_{side}"] = ddy
    df[f"curvature_{side}"] = kappa

    print(f"✅ 已寫入: dy_{side}, ddy_{side}, curvature_{side}")
    print(f"📊 curvature_{side} 統計: min={np.min(kappa):.4f}, max={np.max(kappa):.4f}, mean={np.mean(kappa):.4f}")

def find_peaks_between_valleys(data, valley_indices, bar, min_height_diff=8, verbose=True):
    """
    在每段 valley 之間找出中位數波峰（median_peaks）與所有波峰的 index（absolute_peaks_list）。
    僅對第一段與最後一段的 peaks（含 fallback）使用 bar 檢查靜止區間（bar 變化小於 3 即排除）。
    
    回傳：
        median_peaks: 每段的代表波峰（長度 = len(valley_indices)+1）
        absolute_peaks_list: 所有找到的波峰 index
    """
    median_peaks = []
    absolute_peaks_list = []

    extended_valleys = [-1] + valley_indices + [len(data)]  # 虛擬起點與終點補齊
    num_segments = len(extended_valleys) - 1

    for i in range(num_segments):
        v_start = 0 if extended_valleys[i] == -1 else extended_valleys[i]
        v_end = len(data) if extended_valleys[i + 1] == len(data) else extended_valleys[i + 1]

        segment = data[v_start:v_end]
        if len(segment) == 0:
            continue

        relative_peaks, _ = find_peaks(segment)
        filtered_relative_peaks = []

        for rp in relative_peaks:
            global_idx = v_start + rp
            peak_val = segment[rp]

            # 👉 僅在最前段與最後一段使用 bar 過濾
            if i == 0 or i == num_segments - 1:
                bar_window = bar[max(global_idx - 5, 0):min(global_idx + 5, len(bar))]
                if len(bar_window) >= 2 and (np.max(bar_window) - np.min(bar_window)) < 3:
                    continue  # 跳過靜止區域

            if (peak_val - segment[0] >= min_height_diff) and (peak_val - segment[-1] >= min_height_diff):
                filtered_relative_peaks.append(rp)

        if filtered_relative_peaks:
            absolute_peaks = [v_start + rp for rp in filtered_relative_peaks]
            absolute_peaks.sort()
            absolute_peaks_list.extend(absolute_peaks)
            median_peak = absolute_peaks[len(absolute_peaks) // 2]
            median_peaks.append(median_peak)

            if verbose:
                print(f"第 {i+1} 段有效 peaks: {absolute_peaks}, 中位數 peak: {median_peak}")
        else:
            # === Fallback：找 bar 有移動的最大點 ===
            fallback_candidates = np.argsort(segment)[::-1]  # 從大到小排序 index（相對於 segment）
            fallback_peak = None

            for idx in fallback_candidates:
                global_idx = v_start + idx
                if i == 0 or i == num_segments - 1:
                    bar_window = bar[max(global_idx - 5, 0):min(global_idx + 5, len(bar))]
                    if len(bar_window) >= 2 and (np.max(bar_window) - np.min(bar_window)) < 3:
                        continue  # 跳過 bar 沒動的 fallback
                fallback_peak = global_idx
                break  # 找到符合 bar 條件的最高點

            if fallback_peak is None:
                fallback_peak = v_start + np.argmax(segment)  # 保底策略：直接取最大值（bar 靜止也用）

            median_peaks.append(fallback_peak)
            absolute_peaks_list.append(fallback_peak)

            if verbose:
                print(f"第 {i+1} 段無有效 peak，使用 fallback: {fallback_peak}")

    return median_peaks, absolute_peaks_list

#0707
def find_motion_segments(df):
    """
    使用 valley 與曲率分析找出每段動作的 start 與 end 點
    - valley: 動作最低點（主節點）
    - 曲率右側最小點 → 作為 start
    - 曲率左側最小點 → 作為 end
    - 若首段或尾段無法找到對應點，則補點
    回傳：
        segments: List of (start_frame, valley_frame, end_frame)
        valley_indices: List of valley index (原始 index)
    """
    _, _, valley_indices = analyze_valleys_peaks(df)
    angle = df["left_elbow_angle_smoothing"].values
    bar = df["smoothing_uniform_1"].values
    kappa_org = df["curvature_left_x4"].values
    kappa = df["curvature_left_x4_smoothing"].values

    segments = []

    # 🎯 取得每段 valley 間的虛擬 peak 作為曲率搜尋基準
    peaks, absolute_peaks_list = find_peaks_between_valleys(angle, valley_indices, bar)
    print(f"peaks is {peaks}")
    left_corners, right_corners = find_corners_by_curvature_with_start_end(kappa_org, angle, valley_indices, peaks)

    start_list = right_corners.copy()  # 起點：右鈍角
    end_list = left_corners.copy()    # 終點：左鈍角

    print(f"✅ 曲率轉折點分析完畢: start_list={start_list}, end_list={end_list}")

    # === 組合段落 ===
    if not (len(start_list) == len(valley_indices) == len(end_list)):
        print("❌ 長度不一致：start / valley / end =", len(start_list), len(valley_indices), len(end_list))
        return [], valley_indices

    for s, v, e in zip(start_list, valley_indices, end_list):
        segments.append((
            df.loc[s, "Frame"],
            df.loc[v, "Frame"],
            df.loc[e, "Frame"]
        ))

    return segments, valley_indices, peaks, absolute_peaks_list




def export_plot(df, plot_file, segments, base_path, valley_indices, peaks, absolute_peaks_list):
    fig, ax1 = plt.subplots(figsize=(14, 6))  # 建立主軸

    # 畫主要角度線條
    ax1.set_ylabel("Angle (degrees)", fontsize=12)
    ax1.set_xlabel("Frame", fontsize=12)

    # 主軸：畫 smoothing 左右手肘角度
    if "left_elbow_angle_smoothing" in df.columns or "right_elbow_angle_smoothing" in df.columns:
        if "left_elbow_angle_smoothing" in df.columns:
            ax1.plot(df["Frame"], df["left_elbow_angle_smoothing"], label="left elbow", linewidth=1.5, color='blue')  # 主軸左肘
        # if "right_elbow_angle_smoothing" in df.columns:
        #     ax1.plot(df["Frame"], df["right_elbow_angle_smoothing"], label="right_elbow_angle_smoothing", linewidth=1.5, color='orange')  # 主軸右肘
        if "smoothing_uniform_1_d3" in df.columns:
            ax1.plot(df["Frame"], df["smoothing_uniform_1_d3"], label="bar position", linewidth=1.5, color='orange')  # 主軸左肘
        ax1.set_ylabel("Elbow Smoothing Angles", fontsize=12)
   
    # 副軸：畫 dy、ddy、curvature
    has_secondary = any(col in df.columns for col in ["dy_left", "ddy_left", "curvature_left"])
    if has_secondary:
        ax2 = ax1.twinx()
        # if "dy_left" in df.columns:
        #     ax2.plot(df["Frame"], df["dy_left_d4"], label="dy_left_d4", linestyle="--", linewidth=1.2, color='green')
        # if "ddy_left" in df.columns:
        #     ax2.plot(df["Frame"], df["ddy_left"], label="ddy_left", linestyle="--", linewidth=1.2, color='purple')
        if "curvature_left" in df.columns: 
            ax2.plot(df["Frame"], df["curvature_left_x4"], label="curvature_left_x4", linestyle="--", linewidth=1.2, color='brown')
            #ax2.plot(df["Frame"], df["curvature_left_x4_smoothing"], label="curvature_left_x4_smoothing", linestyle="--", linewidth=1.2, color='red')
        ax2.set_ylabel("dy / ddy / curvature", fontsize=12)
        # if "curvature_right" in df.columns:
        #     ax2.plot(df["Frame"], df["curvature_right_x4"], label="curvature_right", linestyle="--", linewidth=1.2, color='green')

    else:
        ax2 = None

    # 加上圖例（合併左右軸）
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", ncol=4)

    ax1.grid(True)
    ax1.set_title("Joint Angles and Action Point", fontsize=16)

    first_start = True
    first_end = True
    print("DEBUG: segments =", segments)
    print("DEBUG: type =", type(segments), "length =", len(segments) if segments else 0)

    for seg in segments:
        start_frame = seg[0]
        end_frame = seg[2]
        start_match = df[df["Frame"] == start_frame]
        end_match = df[df["Frame"] == end_frame]
        if not start_match.empty:
            y_val = start_match["left_elbow_angle_smoothing"].values[0]
            ax1.scatter(start_frame, y_val, color="red", s=40, zorder=5, label="Start Point" if first_start else None)
            first_start = False

        if not end_match.empty:
            y_val = end_match["left_elbow_angle_smoothing"].values[0]
            offset = 6
            ax1.scatter(end_frame, y_val + offset, color="blue", s=40, zorder=5, label="End Point" if first_end else None)
            first_end = False
    first_valley = True
    for valley in valley_indices:
        match = df[df["Frame"] == valley]
        if not match.empty:
            y_val = match["left_elbow_angle_smoothing"].values[0]
            offset = 3  # 若你想讓 valley 畫高一點（避免重疊），可加 offset
            ax1.scatter(valley, y_val, color="black", s=40, zorder=5, label="Valley" if first_valley else None)
            first_valley = False

    for peak in absolute_peaks_list:
        match = df[df["Frame"] == peak]
        if not match.empty:
            y_val = match["left_elbow_angle_smoothing"].values[0]
            ax1.scatter(peak, y_val + offset, color="yellow", s=50, marker='o', zorder=5)
            first_peak = False


        # 標註 peak 點（用 ▼ 表示）
    first_peak = True
    for peak in peaks:
        match = df[df["Frame"] == peak]
        if not match.empty:
            y_val = match["left_elbow_angle_smoothing"].values[0]
            ax1.scatter(peak, y_val + offset, color="darkgreen", s=50, marker='v', zorder=5, label="Peak" if first_peak else None)
            first_peak = False

    # ➤ 最後才加圖例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", ncol=4)
    #為了比照所有圖片所加-------------------------------------------------------
 # ➤ 最後儲存兩次圖片（一次原始，一次指定目錄）
    plt.tight_layout()
    plt.savefig(plot_file)

    # 如果要存另一份，要確保畫布還在
    if os.path.exists(plot_file):
        print(f"DEBUG: Original plot saved successfully at {plot_file}")
    else:
        print(f"DEBUG: Failed to save original plot at {plot_file}")

    plt.close()  # 最後再關閉畫布


def export_excel_plot_only(df, output_file):
    print("📤 [Step 1] 開始匯出 Excel 到:", output_file)

    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet("Plot")

    # 檢查欄位是否存在
    required_columns = [
        "Frame", "Left Elbow Angle", "Right Elbow Angle",
        "Left Shoulder Angle", "Right Shoulder Angle",
        "avg_elbow_angle", "smoothing_uniform_1", "avg_elbow_angle_smoothing", 
        "right_elbow_angle_smoothing", "left_elbow_angle_smoothing","dy_left", "dy_right","ddy_left", "ddy_right","curvature_left",
        "dy_left_d4", "curvature_left_x4", "dy_right_d4", "curvature_right_x4", "curvature_left_x4_smoothing", "smoothing_uniform_1_d3",
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 缺少必要欄位: {col}")
            workbook.close()
            return

    # ➤ 寫入標題與資料
    print("✍️ [Step 2] 寫入資料")
    headers = df.columns.tolist()
    for col_idx, header in enumerate(headers):
        worksheet.write(0, col_idx, header)
        for row_idx, value in enumerate(df[header]):
            worksheet.write(row_idx + 1, col_idx, value)
    print("✅ 寫入完成")

    # ➤ 建立圖表
    print("📈 [Step 3] 建立圖表")
    chart = workbook.add_chart({'type': 'line'})

    # 主軸欄位與顏色
    primary_columns = [
        ("smoothing_uniform_1_d3", 'blue'),
        ("left_elbow_angle_smoothing", 'green'),
        #("Left Shoulder Angle", 'orange'),
        #("Right Shoulder Angle", 'purple'),
        #("avg_elbow_angle", 'black'),
        #("avg_elbow_angle_smoothing", 'red')  # 主軸顯示 smoothing 線
    ]

    for name, color in primary_columns:
        col_idx = df.columns.get_loc(name)
        print(f"➕ 加入主軸線: {name}（顏色: {color}，欄位: {col_idx}）")
        chart.add_series({
            'name':       [worksheet.name, 0, col_idx],
            'categories': [worksheet.name, 1, 0, len(df), 0],
            'values':     [worksheet.name, 1, col_idx, len(df), col_idx],
            'line':       {'color': color, 'width': 1.5}
        })

    # 副軸欄位（導數與曲率）
    secondary_columns = [
        ("curvature_left_x4", 'brown'),
        ("curvature_left_x4_smoothing", 'magenta')
        #,("curvature_left", 'gray')
    ]

    for name, color in secondary_columns:
        col_idx = df.columns.get_loc(name)
        print(f"➕ 加入副軸線: {name}（顏色: {color}，欄位: {col_idx}）")
        chart.add_series({
            'name':       [worksheet.name, 0, col_idx],
            'categories': [worksheet.name, 1, 0, len(df), 0],
            'values':     [worksheet.name, 1, col_idx, len(df), col_idx],
            'y2_axis':    True,
            'line':       {'color': color, 'dash_type': 'dash', 'width': 1.5}
        })

    chart.set_title({'name': 'Angle Analysis'})
    chart.set_x_axis({'name': 'Frame'})
    chart.set_y_axis({'name': 'Angle (degrees)'})
    chart.set_y2_axis({'name': 'dy / ddy / curvature'})
    chart.set_legend({'position': 'top'})

    # ➤ 插入圖表
    print("📌 插入圖表到欄位 L2")
    worksheet.insert_chart('L2', chart)

    workbook.close()
    print("✅ 匯出完成，請開啟檢查圖表位置")


def bar_data_processing(input_file_bar):  # 處理barbell資料，反轉X/Y座標並只留下Frame與Y軸位置
    data = pd.read_csv(input_file_bar, header=None)  # 讀取無標頭的CSV檔案
    data.columns = ["Frame", "X-axis-location", "Y-axis-location", "width", "height"]  # 指定欄位名稱
    
    # 座標反轉（這裡假設畫面大小為 640x480，可依實際資料修改）
    data["X-axis-location"] = 600 - data["X-axis-location"]  # X 座標反轉
    data["Y-axis-location"] = 600 - data["Y-axis-location"]  # Y 座標反轉

    return data[["Frame", "Y-axis-location"]].rename(columns={"Y-axis-location": "y-axis"})  # 回傳frame與反轉後Y軸，並改名


#segment儲存與切割
def save_segments_txt(segments, output_file):  # 將切割段落存成文字檔
    with open(output_file, "w") as f:
        for seg in segments:
            f.write(f"{seg[0]} - {seg[2]}\n")  # 寫入起中止 Frame-{seg[1]}-
    print(segments)  # 印出段落清單

def split_segments_txt_to_individual_files(input_file, output_dir):  # 將每段存成獨立 txt 檔案
    # 🔹 先刪除所有 P{n}.txt 檔案
    for file in os.listdir(output_dir):  # 列出輸出資料夾內所有檔案
        if re.match(r"cut4\d+\.txt", file):  # 正則比對符合 P1.txt、P2.txt 等
            os.remove(os.path.join(output_dir, file))  # 刪除舊檔案

    with open(input_file, "r") as f:
        lines = f.readlines()  # 讀取所有段落行

    for idx, line in enumerate(lines, start=1):  # 從 P1 開始命名
        line = line.strip()
        if not line:
            continue
        filename = os.path.join(output_dir, f"cut4_{idx}.txt")  # 組成輸出檔案路徑
        with open(filename, "w") as out_file:
            out_file.write(line)  # 寫入分段內容



def compute_total_temporal_error(base_path):  # 計算每組自動切段與手動切段的誤差（起點與終點）
    start_diffs = []  # 起始點誤差
    end_diffs = []    # 結束點誤差

    # 尋找所有 P{n}.txt 檔案
    p_files = [f for f in os.listdir(base_path) if re.match(r"P\d+\.txt", f)]  # 找出所有符合格式的自動切段檔案
    p_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))  # 根據數字排序

    print("======= 每組誤差結果 =======")

    for p_file in p_files:
        match = re.search(r"P(\d+)\.txt", p_file)  # 擷取編號
        if not match:
            continue
        n = int(match.group(1))  # 編號 n
        manual_file = os.path.join(base_path, f"cut4_{n}.txt")  # 手動檔案
        auto_file = os.path.join(base_path, p_file)  # 自動檔案

        if not os.path.exists(manual_file):
            print(f"⚠️ 找不到 cut4_{n}.txt，跳過此組。")
            continue

        try:
            with open(manual_file, "r") as f1:
                manual_line = f1.readline().strip()  # 讀取手動切段
                numbers = list(map(int, re.findall(r"\d+", manual_line)))  # 抓出所有數字
                if len(numbers) != 2:
                    raise ValueError(f"⚠️ 格式錯誤，無法解析: {manual_line}")
                manual_start, manual_end = numbers  # 手動起訖點

            with open(auto_file, "r") as f2:
                auto_line = f2.readline().strip()  # 讀取自動切段
                auto_start, auto_end = map(int, auto_line.split(" - "))  # 拆分起訖點

            start_error = abs(auto_start - manual_start)  # 起點誤差
            end_error = abs(auto_end - manual_end)        # 終點誤差

            print(f"[組 {n:>2}] 手動: ({manual_start}, {manual_end})｜自動: ({auto_start}, {auto_end})｜start_diff={start_error}, end_diff={end_error}")
            start_diffs.append(start_error)  # 收集起點誤差
            end_diffs.append(end_error)      # 收集終點誤差

        except Exception as e:
            print(f"⚠️ 無法處理第{n}組: {e}")

    if not start_diffs or not end_diffs:
        print("❌ 沒有任何有效的資料對。")
        return

    start_std = np.std(start_diffs)  # 起點誤差標準差
    end_std = np.std(end_diffs)      # 終點誤差標準差
    avg_start_error = np.mean(start_diffs)  # 起點誤差平均值
    avg_end_error = np.mean(end_diffs)      # 終點誤差平均值

    print("\n======= 結果總結 =======")
    print(f"起點誤差平均值: {avg_start_error:.2f} frame")
    print(f"起點誤差標準差: {start_std:.2f} frame")
    print(f"終點誤差平均值: {avg_end_error:.2f} frame")
    print(f"終點誤差標準差: {end_std:.2f} frame")
    print(f"有效資料組數: {len(start_diffs)}")

    return {
        "start_std": start_std,
        "end_std": end_std,
        "avg_start_error": avg_start_error,
        "avg_end_error": avg_end_error,
        "pairs": len(start_diffs)
    }




#合併切割frame做成影片
def create_merged_video_with_labels(video_path, segments_file, output_file):
    import cv2
    import os
    import re

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 讀取分段資訊
    with open(segments_file, "r") as f:
        lines = f.readlines()

    segment_infos = []
    for idx, line in enumerate(lines, start=1):
        numbers = list(map(int, re.findall(r"\d+", line)))
        if len(numbers) == 2:
            start_frame, end_frame = numbers
            segment_infos.append((idx, start_frame, end_frame))

    if not segment_infos:
        print("❌ 沒有有效的分段資訊。")
        return

    #--- 逐個段落處理 ---
    for seg_idx, start_frame, end_frame in segment_infos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 移動到該段的起始 frame

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ 無法讀取 frame {frame_idx}，跳過。")
                break

            # ======= Counts: 顯示在右下角，黑底白字 =======
            counts_text = f"Counts: {seg_idx}"
            (counts_width, counts_height), _ = cv2.getTextSize(counts_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            counts_x = width - counts_width - 20
            counts_y = height - 20

            # 畫黑底矩形
            cv2.rectangle(frame,
                          (counts_x - 5, counts_y - counts_height - 5),
                          (counts_x + counts_width + 5, counts_y + 5),
                          (0, 0, 0),
                          thickness=-1)  # 填滿黑色
            # 畫白字
            cv2.putText(frame, counts_text, (counts_x, counts_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2)

            # ======= Frame: 顯示在左下角，黑底白字 =======
            frame_text = f"Frame {frame_idx}"
            (frame_width, frame_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            frame_x = 20
            frame_y = height - 20

            # 畫黑底矩形
            cv2.rectangle(frame,
                          (frame_x - 5, frame_y - frame_height - 5),
                          (frame_x + frame_width + 5, frame_y + 5),
                          (0, 0, 0),
                          thickness=-1)
            # 畫白字
            cv2.putText(frame, frame_text, (frame_x, frame_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(255, 255, 255),
                        thickness=2)

            # ======= 輸出這一幀 =======
            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 完成合併影片：{output_file}")

def merge_skeleton_and_bar(df_skeleton, df_bar):
    df_bar = df_bar.rename(columns={"y-axis": "bar_y_axis"})
    df_merged = pd.merge(df_skeleton, df_bar, on="Frame", how="left")
    return df_merged

def analyze_elbow_motion(input_file, input_file_bar, output_plot, output_excel, output_txt, base_path):
    df = parse_frame_data(input_file)

    # 若有 bar 資料則合併
    if input_file_bar and os.path.exists(input_file_bar):
        df_bar = bar_data_processing(input_file_bar)
        df = merge_skeleton_and_bar(df, df_bar)

    #----------0523add
    df["smoothing_uniform_1"] = apply_uniform_smoothing(df["bar_y_axis"])
    df["smoothing_uniform_1"] = fill_edges(df["smoothing_uniform_1"])

    # df["smoothing_uniform_2"] = apply_uniform_smoothing(df["smoothing_uniform_1"])
    # df["smoothing_uniform_2"] = fill_edges(df["smoothing_uniform_2"])

    # df["smoothing_uniform_3"] = apply_uniform_smoothing(df["smoothing_uniform_2"])
    # df["smoothing_uniform_3"] = fill_edges(df["smoothing_uniform_3"])

    df["avg_elbow_angle_smoothing"] = apply_rolling_smoothing(df["avg_elbow_angle"])
    df["avg_elbow_angle_smoothing"] = fill_edges(df["avg_elbow_angle_smoothing"])

    df["left_elbow_angle_smoothing"] = apply_rolling_smoothing(df["Left Elbow Angle"])
    df["left_elbow_angle_smoothing"] = fill_edges(df["left_elbow_angle_smoothing"])
    df["right_elbow_angle_smoothing"] = apply_rolling_smoothing(df["Right Elbow Angle"])
    df["right_elbow_angle_smoothing"] = fill_edges(df["right_elbow_angle_smoothing"])
    #0523add end
    compute_derivatives(df, "left")
    compute_derivatives(df, "right")
    #segments , valley_indices = find_motion_segments(df)

    df["dy_left_d4"] = df["dy_left"] / 4
    df["curvature_left_x4"] = df["curvature_left"] * 3
    df["dy_right_d4"] = df["dy_right"] / 4
    df["curvature_right_x4"] = df["curvature_right"] * 3
    df["smoothing_uniform_1_d3"] = df["smoothing_uniform_1"] / 3

    df["curvature_left_x4_smoothing"] = apply_rolling_smoothing(df["curvature_left_x4"])
    df["curvature_left_x4_smoothing"] = fill_edges(df["curvature_left_x4_smoothing"])
    segments , valley_indices , peaks, absolute_peaks_list= find_motion_segments(df)


    
    export_plot(df, output_plot, segments, base_path, valley_indices, peaks, absolute_peaks_list)
    #export_excel(df, segments, output_excel)
    #-----------0523add
    # ✅ 儲存到 Excel
    df.to_excel(output_excel, index=False)
    print(f"✅ 已更新並儲存三階段平滑欄位至 {output_excel}")
    export_excel_plot_only(df, output_excel)

     #..............add end
    save_segments_txt(segments, output_txt)
    output_dir = os.path.dirname(output_txt)
    split_segments_txt_to_individual_files(output_txt, output_dir)

    # diff_result = compute_total_temporal_error(base_path)
    # print(f"✅ 分析完成，總共偵測到 {len(segments)} 段動作。")
    # return diff_result
    print(f"✅ 分析完成，總共偵測到 {len(segments)} 段動作。")
    return

class StartColResolver:
    def __init__(self):
        self.category_to_startcol = {
            "正常": 0,
            "右邊低": 6,
            "左邊低": 12,
            "肩胛骨往上": 18,
            "肩膀": 24,
            "壓手腕": 30, 
            "右邊低_壓手腕": 36,
            "左邊低_壓手腕": 42,
            "肩胛骨往上_壓手腕":48 ,
            "肩膀_壓手腕": 54
        }

    def get_category_from_path(self, path):
        # 拆解資料夾結構，取倒數第二層
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 2:
            raise ValueError("路徑層級不足，無法擷取倒數第二層資料夾")
        return parts[-2]

    def get_startcol(self, path):
        category = self.get_category_from_path(path)
        if category not in self.category_to_startcol:
            raise ValueError(f"未知的分類名稱: {category}")
        return self.category_to_startcol[category]

#--------------------------------------------
# 📌 第一階段：rolling 平滑
def apply_rolling_smoothing(series, window=12):
    return series.rolling(window=window, center=True).mean()

# 📌 第二階段：Hampel 平滑
def apply_hampel_filter(series, window_size=15, n_sigmas=45):
    series = series.copy()
    L = 1.4826
    rolling_median = series.rolling(window_size, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = L * diff.rolling(window_size, center=True).median()
    outliers = diff > n_sigmas * mad
    series[outliers] = rolling_median[outliers]
    return series

# 📌 第三階段：均值濾波 uniform_filter1d
# def apply_uniform_smoothing(series, window_size=9):
#     filled_series = series.fillna(method="ffill").fillna(method="bfill")  # 前向補完後再補後向
#     smoothed = uniform_filter1d(filled_series, size=window_size)
#     return pd.Series(smoothed, index=series.index)
def apply_uniform_smoothing(series, window_size=7, shift_size=2):
    filled_series = series.ffill().bfill()  # 前向後向補值
    smoothed = uniform_filter1d(filled_series.values, size=window_size)  # 均值濾波

    shift = shift_size // 2  # 向左補正
    shifted = np.roll(smoothed, -shift)  # 向左平移

    # 邊界補回 NaN（因為 np.roll 是環狀）
    shifted[-shift:] = np.nan

    return pd.Series(shifted, index=series.index)


# 📌 儲存或更新欄位
def update_or_append_column(df, column_name, new_data):
    df[column_name] = new_data  # 直接更新或新增
    return df

def fill_edges(series):
    # 中間插值 + 開頭與結尾向外延伸填補
    return series.interpolate(method="linear").bfill().ffill()  # 插值後向後補，再向前補



if __name__ == "__main__":                                                                                         # 入口
    import sys, os                                                                                                  # 匯入基礎模組
    results = []                                                                                                    # 結果容器（若上層需要彙整可用）

    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                        # 預設 recordings 根目錄
    # 用法範例（UI 會這樣呼叫）：python this_script.py C:\Users\92A27\benchpress\recordings\recording_YYYYMMDD_HHMMSS   # 說明

    if len(sys.argv) >= 2:                                                                                          # 有傳入資料夾參數
        base_path = sys.argv[1]                                                                                     # 取第一個參數
        if not os.path.isdir(base_path):                                                                            # 檢查是否為有效資料夾
            raise FileNotFoundError(f"❌ 指定的資料夾不存在：{base_path}")                                             # 拋錯
    else:                                                                                                           # 沒有傳參數 → 退回找最新
        if not os.path.isdir(recordings_dir):                                                                       # recordings 根目錄要存在
            raise FileNotFoundError(f"❌ recordings 根目錄不存在：{recordings_dir}")                                   # 拋錯
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)                                 # 列出子資料夾
                if os.path.isdir(os.path.join(recordings_dir, d))]                                                  # 過濾出資料夾
        if not subs:                                                                                                # 沒有任何子資料夾
            raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾，且未提供參數")                              # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                                 # 取最後修改時間最新的資料夾

    # --- 路徑組裝統一用 os.path.join，避免手動加斜線帶來的錯誤 ---                                                     # 說明
    input_file = os.path.join(base_path, "yolo_skeleton_interpolated_hampel.txt")                                   # 骨架檔
    input_file_bar = os.path.join(base_path, "yolo_coordinates_interpolated_hampel.txt")                            # 槓端檔
    output_plot = os.path.join(base_path, "elbow_shoulder_angles_plot.png")                                         # 輸出圖
    output_excel = os.path.join(base_path, "peak_analysis.xlsx")                                                     # 輸出 Excel
    output_txt = os.path.join(base_path, "elbow_shoulder_angles_cut.txt")                                           # 切段結果
    video_file = os.path.join(base_path, "original_vision3.avi")                                                    # 原始影片
    merged_video = os.path.join(base_path, "merged_segments_with_labels.avi")                                       # 合成帶標籤影片

    print(f"▶ 實際處理資料夾：{base_path}")                                                                            # 確認本次處理對象

    # 檢查主要輸入檔案是否存在                                                                                           # 註解
    if not os.path.exists(input_file):                                                                              # 檢查骨架檔
        print(f"❌ 找不到骨架檔案：{input_file}")                                                                     # 警示
        sys.exit(1)                                                                                                  # 結束（回傳非零碼）

    # 呼叫分析函式                                                                                                      # 註解
    result = analyze_elbow_motion(                                                                                  # 執行分析
        input_file=input_file,                                                                                       # 傳入骨架檔
        output_plot=output_plot,                                                                                     # 輸出圖路徑
        output_excel=output_excel,                                                                                   # 輸出 Excel
        output_txt=output_txt,                                                                                       # 輸出切段 txt
        base_path=base_path,                                                                                         # 基底資料夾
        input_file_bar=input_file_bar                                                                               # 槓端檔路徑
    )                                                                                                                # 結束呼叫

    # 呼叫影片組合函式（如果影片存在才執行）                                                                              # 註解
    if os.path.exists(video_file):                                                                                   # 檢查原始影片是否存在
        create_merged_video_with_labels(                                                                             # 合併片段並上標籤
            video_path=video_file,                                                                                   # 輸入影片
            segments_file=output_txt,                                                                                # 切段文字檔
            output_file=merged_video                                                                                 # 合成輸出
        )                                                                                                            # 結束呼叫
    else:                                                                                                            # 影片不存在
        print(f"⚠️ 找不到影片檔案，略過合成：{video_file}")                                                             # 提示

    print(f"✅ 已完成處理：{base_path}")                                                                               # 完成訊息

