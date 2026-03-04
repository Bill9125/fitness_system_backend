import numpy as np
import pandas as pd
import os
import json
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# --- Utility Functions ---

def get_angle(a, b, c):
    """Calculates the angle ABC (at vertex B) in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return np.nan
    angle_deg = np.degrees(np.arccos(np.clip(dot / (norm_ab * norm_cb), -1.0, 1.0)))
    return angle_deg if angle_deg <= 180 else 360 - angle_deg

def apply_rolling_smoothing(series, window=12):
    return series.rolling(window=window, center=True).mean()

def apply_uniform_smoothing(series, window_size=7, shift_size=2):
    filled_series = series.ffill().bfill()
    smoothed = uniform_filter1d(filled_series.values, size=window_size)
    shift = shift_size // 2
    shifted = np.roll(smoothed, -shift)
    shifted[-shift:] = np.nan
    return pd.Series(shifted, index=series.index)

def fill_edges(series):
    return series.interpolate(method="linear").bfill().ffill()

def compute_derivatives(df, side):
    """Compute dy, ddy, curvature for a given side ('left' or 'right')."""
    col_name = f"{side}_elbow_angle_smoothing"
    if col_name not in df.columns:
        return
    angle = df[col_name].values
    dy = np.gradient(angle)
    ddy = np.gradient(dy)
    kappa = ddy / np.power(1 + dy**2, 1.5)
    df[f"dy_{side}"] = dy
    df[f"ddy_{side}"] = ddy
    df[f"curvature_{side}"] = kappa
    # x4 scaled version (used for corner detection, like original)
    df[f"curvature_{side}_x4"] = kappa * 3
    # smoothed version for find_corners
    smoothed = apply_rolling_smoothing(pd.Series(kappa * 3))
    df[f"curvature_{side}_x4_smoothing"] = smoothed.interpolate().bfill().ffill()

# --- Core Logic (ported faithfully from original) ---

def find_clear_difference_side(angle, idx, min_peak_diff, max_search=70):
    """Check if there is a clearly higher point on both sides of idx."""
    val = angle[idx]

    left_idx = idx - 1
    while left_idx >= max(0, idx - max_search):
        if angle[left_idx] - val >= min_peak_diff:
            break
        left_idx -= 1
    else:
        return False

    right_idx = idx + 1
    while right_idx < min(len(angle), idx + max_search):
        if angle[right_idx] - val >= min_peak_diff:
            break
        right_idx += 1
    else:
        return False

    return True


def is_clear_valley(angle, idx, window=25, min_prominence=0.05, future_window=18,
                    slope_check=True, min_peak_diff=100, is_first=False):
    """Full valley validation with 5 conditions, matching the original."""
    idx = int(idx)
    val = angle[idx]

    start = max(0, idx - window)
    end = min(len(angle), idx + window)
    left = angle[start:idx]
    right = angle[idx+1:end]

    if len(left) == 0 or len(right) == 0:
        print(f"[{idx}] ⚠️ 無法取得足夠左右資料")
        return False

    left_max = np.max(left)
    right_max = np.max(right)

    # Condition 1: Valley shape prominence
    dynamic_prominence = min(min_prominence * val, 20)
    condition_valley_shape = (
        val < left_max and
        val < right_max and
        (left_max - val > dynamic_prominence) or
        (right_max - val > dynamic_prominence)
    )
    if not condition_valley_shape:
        print(f"[{idx}] ❌ shape 不成立: val={val:.2f}, left_max={left_max:.2f}, right_max={right_max:.2f}")

    # Condition 2: No further drop in next future_window frames
    future = angle[idx+1 : idx+1+future_window]
    condition_no_later_drop = all(val <= f for f in future)
    if not condition_no_later_drop:
        print(f"[{idx}] ❌ 未來有下降: val={val:.2f}, future={future[:5]}...")

    # Condition 3: Clear difference on both sides
    condition_peak_diff = find_clear_difference_side(angle, idx, min_peak_diff)

    # Condition 4: Pre-slope must be descending
    condition_pre_slope = True
    past_window = 20
    if slope_check and idx - past_window >= 0:
        prev = angle[idx - past_window:idx]
        mean_pre_slope = np.mean(np.diff(prev))
        condition_pre_slope = mean_pre_slope < 0
        if not condition_pre_slope:
            print(f"[{idx}] ❌ 前段斜率不符: mean_pre_slope={mean_pre_slope:.4f}")

    # Condition 5: Post-slope must be ascending
    condition_post_slope_tail = True
    if slope_check and idx + past_window < len(angle):
        post = angle[idx + 1:idx + 1 + past_window]
        mean_post_slope = np.mean(np.diff(post))
        condition_post_slope_tail = mean_post_slope > 0
        if not condition_post_slope_tail:
            print(f"[{idx}] ❌ 後段斜率不符: mean_post_slope={mean_post_slope:.4f}")

    return (
        condition_valley_shape and
        condition_no_later_drop and
        condition_peak_diff and
        condition_pre_slope and
        condition_post_slope_tail
    )


def filter_close_valleys(angle, valley_indices, peak_indices, min_peak_diff=60):
    """Merge adjacent valleys if no significant peak between them (false double-valley)."""
    filtered_valleys = []
    i = 0
    while i < len(valley_indices):
        current_valley = valley_indices[i]
        if i + 1 < len(valley_indices):
            next_valley = valley_indices[i + 1]
            middle_peaks = [p for p in peak_indices if current_valley < p < next_valley]
            deeper_valley = current_valley if angle[current_valley] < angle[next_valley] else next_valley

            is_all_peaks_shallow = True
            for p in middle_peaks:
                if angle[p] - angle[deeper_valley] >= min_peak_diff:
                    is_all_peaks_shallow = False
                    break

            if is_all_peaks_shallow:
                filtered_valleys.append(deeper_valley)
                i += 2
                continue

        filtered_valleys.append(current_valley)
        i += 1
    return filtered_valleys


def analyze_valleys_peaks(angle_series, min_peak_diff=60):
    """
    Full two-stage valley detection, matching the original analyze_valleys_peaks.
    Input: angle_series - pandas Series of bar_y_axis_smoothing
    """
    angle = angle_series.astype(float).values
    inverted_angle = -angle

    # Bar position threshold
    valid_tail = angle_series.dropna().values[-5:]
    bar_position = np.mean(valid_tail) + 20
    print(f"📏 bar_position 門檻值: {bar_position:.2f}")

    valley_indices_raw, _ = find_peaks(inverted_angle, distance=25, prominence=0.1, width=2)
    print(f"原始 valley_indices_all: {valley_indices_raw}")

    # Stage 1: basic + bar_position filter
    valley_indices = []
    first_valid_idx = None
    for idx in valley_indices_raw:
        is_first = first_valid_idx is None
        is_valid = is_clear_valley(angle, idx, window=25, min_peak_diff=min_peak_diff, is_first=is_first)
        if is_valid and angle[idx] < bar_position:
            if first_valid_idx is None:
                first_valid_idx = idx
            valley_indices.append(int(idx))
    print(f"✅ 第一階段 valley_indices: {valley_indices}")

    # Raw peaks
    peak_indices_raw, _ = find_peaks(angle, distance=25, prominence=0.1, width=2)

    # Peak filtering
    peak_indices = []
    for p_idx in peak_indices_raw:
        if len(valley_indices) == 0:
            continue
        nearest_valley = min(valley_indices, key=lambda v: abs(v - p_idx))
        diff = angle[p_idx] - angle[nearest_valley]
        if diff >= min_peak_diff:
            peak_indices.append(p_idx)
        else:
            print(f"[{p_idx}] ❌ 假波峰，差值 {diff:.2f} < {min_peak_diff}")
    print(f"過濾後的 peak_indices: {peak_indices}")

    # False double-valley merge
    valley_indices = filter_close_valleys(angle, valley_indices, peak_indices, min_peak_diff=min_peak_diff)
    print(f"假雙谷合併後 valley_indices: {valley_indices}")

    # Stage 2: require peak on at least one side
    print("-------------第二次過濾開始-------------")
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
            print(f"[{v_idx}] ❌ 缺少波峰: left={left_peaks}, right={right_peaks}")
            continue

        left_peak = max(left_peaks)
        right_peak = min(right_peaks)
        left_diff = angle[left_peak] - angle[v_idx]
        right_diff = angle[right_peak] - angle[v_idx]

        if left_diff >= min_peak_diff or right_diff >= min_peak_diff:
            final_valleys.append(v_idx)
        else:
            print(f"[{v_idx}] ❌ 差值不夠: left={left_diff:.2f}, right={right_diff:.2f}")

    valley_indices = final_valleys
    print(f"✅ 最終 valley_indices: {valley_indices}")
    return angle, peak_indices, valley_indices


def find_peaks_between_valleys(data, valley_indices, bar, min_height_diff=8, verbose=True):
    """
    Find representative peaks between valleys.
    Uses bar static-zone filter only for first and last segments.
    Returns: (median_peaks, absolute_peaks_list)
    """
    median_peaks = []
    absolute_peaks_list = []

    extended_valleys = [-1] + valley_indices + [len(data)]
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

            # Only apply bar filter on first and last segments
            if i == 0 or i == num_segments - 1:
                bar_window = bar[max(global_idx - 5, 0):min(global_idx + 5, len(bar))]
                if len(bar_window) >= 2 and (np.max(bar_window) - np.min(bar_window)) < 3:
                    continue

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
            # Fallback: largest point where bar is moving
            fallback_candidates = np.argsort(segment)[::-1]
            fallback_peak = None
            for idx in fallback_candidates:
                global_idx = v_start + idx
                if i == 0 or i == num_segments - 1:
                    bar_window = bar[max(global_idx - 5, 0):min(global_idx + 5, len(bar))]
                    if len(bar_window) >= 2 and (np.max(bar_window) - np.min(bar_window)) < 3:
                        continue
                fallback_peak = global_idx
                break

            if fallback_peak is None:
                fallback_peak = v_start + np.argmax(segment)

            median_peaks.append(fallback_peak)
            absolute_peaks_list.append(fallback_peak)
            if verbose:
                print(f"第 {i+1} 段無有效 peak，使用 fallback: {fallback_peak}")

    return median_peaks, absolute_peaks_list


def find_corners_by_curvature(kappa, angle, valley, peaks, threshold=20):
    """
    Find start/end corners by minimum curvature within peak-valley-peak segments.
    Matches original find_corners_by_curvature_with_start_end logic.
    Returns: (ends, starts) — ends are right corners, starts are left corners
    """
    starts = []
    ends = []

    for i in range(len(valley)):
        left_peak = peaks[i]
        right_peak = peaks[i + 1]
        v = valley[i]

        # start: search in [left_peak, valley) for min curvature point above threshold
        seg_start = np.arange(left_peak, v)
        valid_start = None
        if len(seg_start) > 0:
            kappa_seg = kappa[seg_start]
            for idx in np.argsort(kappa_seg):  # smallest (most negative) curvature first
                p_idx = seg_start[idx]
                if angle[p_idx] - angle[v] >= threshold:
                    valid_start = p_idx
                    break
        starts.append(valid_start if valid_start is not None else left_peak)

        # end: search in [valley, right_peak) in REVERSE (right_peak → valley) for min curvature
        seg_end = np.arange(right_peak, v, -1)   # ← KEY FIX: reversed direction
        valid_end = None
        if len(seg_end) > 0:
            kappa_seg = kappa[seg_end]
            for idx in np.argsort(kappa_seg):
                p_idx = seg_end[idx]
                if angle[p_idx] - angle[v] >= threshold:
                    valid_end = p_idx
                    break
        ends.append(valid_end if valid_end is not None else right_peak)

    return ends, starts  # (ends, starts) to match original return order


# --- Main Entry Point ---

def run_autocutting(video_path, bar_dict=None, top_ske_dict=None):
    """
    Main function to process benchpress motion segments.
    Ported faithfully from autocutting_original.py (analyze_elbow_motion / find_motion_segments).
    """
    print(f"[Autocutting] Starting process in {video_path}")

    # 1. Load Data from in-memory dicts
    # Top view joint mapping: 0:L_SHO, 1:R_SHO, 2:L_HIP, 3:R_HIP, 4:L_ELB, 5:R_ELB, 6:L_WRI, 7:R_WRI
    L_SHO, L_ELB, L_WRI = 0, 4, 6
    R_SHO, R_ELB, R_WRI = 1, 5, 7

    data_rows = []

    if top_ske_dict:
        frames = sorted(top_ske_dict.keys())
        for f in frames:
            kps = top_ske_dict[f]
            pts = [(kps[j], kps[j+1]) for j in range(0, len(kps), 2)]

            if len(pts) < 8:
                continue

            l_elb_ang = get_angle(pts[L_WRI], pts[L_ELB], pts[L_SHO])
            r_elb_ang = get_angle(pts[R_WRI], pts[R_ELB], pts[R_SHO])

            bar_y = np.nan
            if bar_dict and f in bar_dict:
                bar_y = bar_dict[f][1]  # Y is index 1

            data_rows.append({
                "Frame": f,
                "Left Elbow Angle": l_elb_ang if not np.isnan(l_elb_ang) else None,
                "Right Elbow Angle": r_elb_ang if not np.isnan(r_elb_ang) else None,
                "bar_y_axis": bar_y
            })
    else:
        print("[Autocutting] No data dictionaries provided, aborting.")
        return []

    if not data_rows:
        print("[Autocutting] No valid data found.")
        return []

    df = pd.DataFrame(data_rows)
    df["avg_elbow_angle"] = df[["Left Elbow Angle", "Right Elbow Angle"]].mean(axis=1)

    # 2. Smoothing — mirrors analyze_elbow_motion in original
    df["smoothing_uniform_1"] = fill_edges(apply_uniform_smoothing(df["bar_y_axis"]))
    df["avg_elbow_angle_smoothing"] = fill_edges(apply_rolling_smoothing(df["avg_elbow_angle"]))
    df["left_elbow_angle_smoothing"] = fill_edges(apply_rolling_smoothing(df["Left Elbow Angle"]))
    df["right_elbow_angle_smoothing"] = fill_edges(apply_rolling_smoothing(df["Right Elbow Angle"]))

    compute_derivatives(df, "left")
    compute_derivatives(df, "right")

    df["smoothing_uniform_1_d3"] = df["smoothing_uniform_1"] / 3

    # 3. Valley/Peak detection (using bar position series, matching original)
    angle_arr, peak_indices, valley_indices = analyze_valleys_peaks(df["smoothing_uniform_1"])

    if not valley_indices:
        print("[Autocutting] No cycles detected.")
        return []

    # 4. Find representative inter-valley peaks
    left_angle = df["left_elbow_angle_smoothing"].values
    bar_vals = df["smoothing_uniform_1"].values
    kappa_org = df["curvature_left_x4"].values      # scaled, unsmoothed — for find_corners
    # kappa_smooth = df["curvature_left_x4_smoothing"].values  # smoothed — not used in corner search

    peaks, absolute_peaks_list = find_peaks_between_valleys(left_angle, valley_indices, bar_vals)
    print(f"peaks is {peaks}")

    # 5. Find start/end corners via curvature
    end_list, start_list = find_corners_by_curvature(kappa_org, left_angle, valley_indices, peaks)
    print(f"✅ start_list={start_list}, end_list={end_list}")

    if not (len(start_list) == len(valley_indices) == len(end_list)):
        print(f"❌ 長度不一致: start={len(start_list)}, valleys={len(valley_indices)}, end={len(end_list)}")
        return []

    # 6. Build segments
    segments = []
    split_info = {}

    config_dir = os.path.join(video_path, "config")
    os.makedirs(config_dir, exist_ok=True)

    for i, (s, v, e) in enumerate(zip(start_list, valley_indices, end_list)):
        s_f = int(df.loc[s, "Frame"])
        v_f = int(df.loc[v, "Frame"])
        e_f = int(df.loc[e, "Frame"])
        segments.append((s_f, v_f, e_f))
        split_info[str(i)] = {"start": s_f, "end": e_f}

    output_json = os.path.join(config_dir, "Split_info.json")
    with open(output_json, "w") as f_json:
        json.dump(split_info, f_json, indent=4)

    print(f"✅ Autocutting complete. Found {len(segments)} segments. Saved to: {output_json}")
    return segments


if __name__ == "__main__":
    pass