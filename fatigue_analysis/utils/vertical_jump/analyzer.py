import numpy as np
import copy

# COCO keypoint indices for Hip
LEFT_HIP = 11

def smooth_series(values, valid_mask=None, window=9):
    """
    Apply centered local polynomial smoothing to a series of values.
    This preserves event timing better than a simple moving average.
    """
    n = len(values)
    if n == 0:
        return []
    
    win = max(1, int(window))
    if win == 1:
        return [float(v) for v in values]
    if win % 2 == 0:
        win += 1

    if valid_mask is None:
        valid_mask = [True] * n

    poly_order = 2
    half = win // 2
    x_all = np.arange(n, dtype=float)
    values_arr = np.array(values, dtype=float)
    valid_arr = np.array(valid_mask, dtype=bool)
    smoothed = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        idx = np.where(valid_arr[lo:hi])[0] + lo
        
        if len(idx) < 2:
            smoothed.append(float(values_arr[i]))
            continue

        order = min(poly_order, len(idx) - 1)
        x_local = x_all[idx] - x_all[i]
        y_local = values_arr[idx]
        try:
            coeffs = np.polyfit(x_local, y_local, order)
            smoothed.append(float(np.polyval(coeffs, 0.0)))
        except np.linalg.LinAlgError:
            smoothed.append(float(np.mean(y_local)))
            
    return smoothed

def interpolate_zero_crossing(xs, ys, lo, hi, mode="best", ref_x=None):
    """
    Linearly interpolate to find the x where y crosses a target (default 0).
    """
    seg_x = np.array(xs[lo:hi + 1], dtype=float)
    seg_y = np.array(ys[lo:hi + 1], dtype=float)
    
    if len(seg_x) < 2:
        return None

    candidates = []
    for i in range(len(seg_y) - 1):
        if seg_y[i] * seg_y[i+1] <= 0: # Crossing zero
            y1, y2 = seg_y[i], seg_y[i+1]
            x1, x2 = seg_x[i], seg_x[i+1]
            if x1 == x2 or y1 == y2:
                frame = (x1 + x2) / 2.0
            else:
                frame = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            candidates.append(frame)

    if not candidates:
        return float(seg_x[np.argmin(np.abs(seg_y))])

    if mode == "first": return candidates[0]
    if mode == "last": return candidates[-1]
    if ref_x is not None:
        return min(candidates, key=lambda x: abs(x - ref_x))
    
    return candidates[0]

def fit_quadratic_apex(xs, ys, coarse_apex_frame, window=2):
    """
    Fit a parabola to the hip Y positions around the coarse apex to find sub-frame accuracy.
    """
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    
    center_idx = np.argmin(np.abs(x_arr - coarse_apex_frame))
    lo = max(0, center_idx - window)
    hi = min(len(x_arr), center_idx + window + 1)
    
    if hi - lo < 3:
        return float(x_arr[center_idx]), float(y_arr[center_idx])

    local_x = x_arr[lo:hi]
    local_y = y_arr[lo:hi]
    
    try:
        a, b, c = np.polyfit(local_x, local_y, 2)
        if a <= 0: # Upside down parabola check (Y is downward in images)
            return float(local_x[np.argmin(local_y)]), float(np.min(local_y))
        apex_x = -b / (2.0 * a)
        apex_y = np.polyval([a, b, c], apex_x)
        return float(apex_x), float(apex_y)
    except:
        return float(local_x[np.argmin(local_y)]), float(np.min(local_y))

def compute_vertical_jump(frames, fps=60.0, gravity=9.81, smooth_win=9):
    """
    Core function to analyze jump height from frame keypoints.
    """
    # 1. Extract raw Hip Y positions
    raw_xs, raw_ys, valid_mask = [], [], []
    for i, frame in enumerate(frames):
        people = frame.get("people", [])
        if not people or len(people[0]["keypoints_xyc"]) <= LEFT_HIP:
            raw_ys.append(0)
            valid_mask.append(False)
        else:
            kpt = people[0]["keypoints_xyc"][LEFT_HIP]
            raw_ys.append(kpt[1])
            valid_mask.append(kpt[2] > 0)
        raw_xs.append(i)

    # 2. Smooth Position
    ys_smoothed = smooth_series(raw_ys, valid_mask, window=smooth_win)
    
    # 3. Derive Velocity (dy/dt)
    vxs = raw_xs[1:]
    vys_raw = [ys_smoothed[i] - ys_smoothed[i-1] for i in range(1, len(ys_smoothed))]
    vys = smooth_series(vys_raw, window=smooth_win) # Smooth velocity
    
    # 4. Derive Acceleration (dv/dt)
    axs = vxs[1:]
    ays = [vys[i] - vys[i-1] for i in range(1, len(vys))]

    # 5. Identify Key Events
    # Coarse apex: v crossing 0 from negative (up) to positive (down)
    # Note: In images Y increases downwards. v < 0 is upward movement.
    v_max_up_idx = np.argmin(vys) # Maximum upward velocity
    v_max_down_idx = np.argmax(vys) # Maximum downward velocity
    
    # Apex (Highest point)
    velocity_zero_frame = interpolate_zero_crossing(vxs, vys, v_max_up_idx, len(vys)-1, mode="first")
    apex_frame, _ = fit_quadratic_apex(raw_xs, ys_smoothed, velocity_zero_frame)
    
    # Takeoff: Acceleration crossing 0 before apex
    takeoff_end_idx = 0
    for i, ax in enumerate(axs):
        if ax <= apex_frame: takeoff_end_idx = i
    takeoff_frame = interpolate_zero_crossing(axs, ays, 0, takeoff_end_idx, mode="last")
    
    # Landing: Acceleration crossing 0 after apex near max downward velocity
    landing_start_idx = 0
    for i, ax in enumerate(axs):
        if ax >= apex_frame:
            landing_start_idx = i
            break
    landing_frame = interpolate_zero_crossing(axs, ays, landing_start_idx, len(axs)-1, mode="best", ref_x=vxs[v_max_down_idx])

    # 6. Calculate Height
    # Flight time from apex to landing (free fall half)
    t_half = (landing_frame - apex_frame) / fps
    jump_height_m = 0.5 * gravity * (t_half ** 2)
    
    return {
        "jump_height_cm": jump_height_m * 100.0,
        "flight_time_sec": t_half * 2, # Total estimated flight time
        "takeoff_frame": round(takeoff_frame, 2),
        "apex_frame": round(apex_frame, 2),
        "landing_frame": round(landing_frame, 2),
    }

if __name__ == "__main__":
    # Example usage with mock data or loading from file
    import json
    import sys
    
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
            results = compute_vertical_jump(data)
            print(f"Jump Height: {results['jump_height_cm']:.2f} cm")
            print(f"Events: {results}")
