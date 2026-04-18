#!/usr/bin/env python3
'''
執行指令(fps60):
/home/yirui/.conda/envs/tcpformer/bin/python /home/yirui/mydata/hip_y_compute.py \
  --batch-dir /home/yirui/mydata/yolo11x/output_jump/FL/jumper_2 \
  --fps 60 \
  --measurement-dir /home/yirui/jump_video/FL/jumper_2/measurement \
  --skip-error-plot \
  --smooth 9

執行指令(fps30):
/home/yirui/.conda/envs/tcpformer/bin/python /home/yirui/mydata/hip_y_compute.py \
  --batch-dir /home/yirui/mydata/yolo11x/output_jump/test_30fps/FL/jumper_2 \
  --fps 30 \
  --measurement-dir /home/yirui/jump_video/test_30fps/FL/jumper_2/measurement \
  --skip-error-plot \
  --smooth 9  


  單一處理指令:
  /home/yirui/.conda/envs/tcpformer/bin/python /home/yirui/mydata/hip_y_compute.py \
  --input /home/yirui/mydata/yolo11x/output_jump/FL/jumper_2/clip_5_keypoints2d.json \
  --output /home/yirui/mydata/yolo11x/output_jump/FL/jumper_2/clip_5_hip_y_pva_overlay.png \
  --title "clip_5 hip Y PVA overlay" \
  --smooth 9 \
  --fps 60
  

  手動輸入幀數指令:
  /home/yirui/.conda/envs/tcpformer/bin/python /home/yirui/mydata/hip_y_compute.py \
  --subject-id jumper_1 \
  --manual-clip-index clip_1 \
  --manual-highest-point-frame 243 \
  --manual-landing-point-frame 259 \
  --fps 60


  直接輸入影片偵測並輸出指令:
  /home/yirui/.conda/envs/tcpformer/bin/python /home/yirui/mydata/hip_y_compute.py \
  --video /home/yirui/jump_video/test_30fps/FL/jumper_1/clip_0.mov \
  --output /home/yirui/mydata/yolo11x/output_jump/test_30fps/FL/jumper_1/clip_0_overlay.mp4 \
  --smooth 9 \
  --fps 30 \
  --yolo-device cpu


'''
import argparse
import copy
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# COCO keypoint indices
LEFT_HIP = 11
RIGHT_HIP = 12

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

STORED_ACTUAL_HEIGHTS_CM = {
    "jumper1": [42.8, 40.9, 40.9, 36.3, 37.8, 40.5, 40.7, 42.8, 43.5, 38.9],
    "jumper2": [48.6, 53.3, 46.5, 54.9, 53.9, 45.2, 49.4, 55.5, 62.1, 55.4],
}


def load_frames(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_pose_frames(video_path: Path, model_path: Path, conf: float, iou: float, device: str):
    from ultralytics import YOLO

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not model_path.exists():
        raise SystemExit(f"YOLO pose model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(video_path),
        stream=True,
        conf=conf,
        iou=iou,
        verbose=False,
        device=device,
    )

    all_frames = []
    for frame_idx, result in enumerate(results):
        frame_data = {"frame_index": frame_idx, "people": []}
        if result.keypoints is not None:
            kpts = result.keypoints.data.cpu().numpy()
            for pid in range(kpts.shape[0]):
                frame_data["people"].append(
                    {
                        "person_id": pid,
                        "keypoints_xyc": kpts[pid].tolist(),
                    }
                )
        all_frames.append(frame_data)
    return all_frames


def smooth_valid_series(values, valid_mask, window: int):
    n = len(values)
    if n == 0:
        return []
    win = max(1, int(window))
    if win == 1:
        return [float(v) for v in values]
    if win % 2 == 0:
        win += 1

    # Centered local polynomial smoothing preserves event timing better
    # than a plain moving average when later taking velocity/acceleration.
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
        if len(idx) == 0:
            smoothed.append(float(values_arr[i]))
            continue
        if len(idx) == 1:
            smoothed.append(float(values_arr[idx[0]]))
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


def smooth_keypoints_frames(frames, window: int):
    if not frames:
        return frames

    max_kpts = 0
    for frame in frames:
        people = frame.get("people", [])
        if people:
            max_kpts = max(max_kpts, len(people[0].get("keypoints_xyc", [])))

    for kpt_idx in range(max_kpts):
        xs = []
        ys = []
        valid_mask = []
        for frame in frames:
            people = frame.get("people", [])
            if not people:
                xs.append(0.0)
                ys.append(0.0)
                valid_mask.append(False)
                continue

            kpts = people[0].get("keypoints_xyc", [])
            if kpt_idx >= len(kpts):
                xs.append(0.0)
                ys.append(0.0)
                valid_mask.append(False)
                continue

            x, y, c = kpts[kpt_idx]
            xs.append(float(x))
            ys.append(float(y))
            valid_mask.append(c > 0)

        if sum(valid_mask) < 2:
            continue

        xs_s = smooth_valid_series(xs, valid_mask, window)
        ys_s = smooth_valid_series(ys, valid_mask, window)
        for frame_idx, frame in enumerate(frames):
            people = frame.get("people", [])
            if not people:
                continue
            kpts = people[0].get("keypoints_xyc", [])
            if kpt_idx >= len(kpts) or not valid_mask[frame_idx]:
                continue
            kpts[kpt_idx][0] = xs_s[frame_idx]
            kpts[kpt_idx][1] = ys_s[frame_idx]

    return frames


def prepare_frames_and_metrics(
    frames,
    smooth_window: int,
    fps: float,
    gravity: float,
    velocity_smooth_window: int,
):
    frames_work = copy.deepcopy(frames)
    frames_work = smooth_keypoints_frames(frames_work, smooth_window)
    metrics = compute_jump_metrics(frames_work, fps, gravity, velocity_smooth_window)
    return frames_work, metrics


def get_hip_y(frame):
    people = frame.get("people", [])
    if not people:
        return None
    kpts = people[0].get("keypoints_xyc", [])
    if len(kpts) <= LEFT_HIP:
        return None

    left_hip = kpts[LEFT_HIP]
    if left_hip[2] <= 0:
        return None
    return float(left_hip[1])


def interpolate_target_frame(xs, ys, lo: int, hi: int, target: float = 0.0, crossing_mode: str = "best"):
    seg_x = np.array(xs[lo:hi + 1], dtype=float)
    seg_y = np.array(ys[lo:hi + 1], dtype=float)
    if len(seg_x) == 0:
        raise SystemExit("No samples available for interpolation.")

    exact_hits = np.where(seg_y == target)[0]
    if exact_hits.size > 0:
        if crossing_mode == "last":
            exact_idx = int(exact_hits[-1])
        else:
            exact_idx = int(exact_hits[0])
        return round(float(seg_x[exact_idx]), 2), float(target)

    best_pair = None
    best_score = None
    for i in range(len(seg_y) - 1):
        y1 = seg_y[i] - target
        y2 = seg_y[i + 1] - target
        if y1 * y2 < 0:
            if crossing_mode == "first":
                best_pair = (i, i + 1)
                break
            if crossing_mode == "last":
                best_pair = (i, i + 1)
                continue
            score = max(abs(y1), abs(y2))
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (i, i + 1)

    if best_pair is None:
        nearest = np.argsort(np.abs(seg_y - target))[:2]
        if len(nearest) < 2:
            only_idx = int(np.argmin(np.abs(seg_y - target)))
            return round(float(seg_x[only_idx]), 2), float(seg_y[only_idx])
        nearest = sorted(int(i) for i in nearest)
        best_pair = (nearest[0], nearest[1])

    i1, i2 = best_pair
    x1 = seg_x[i1]
    x2 = seg_x[i2]
    y1 = seg_y[i1]
    y2 = seg_y[i2]
    if x1 == x2 or y1 == y2:
        target_frame = (x1 + x2) / 2.0
    else:
        target_frame = x1 + ((target - y1) * (x2 - x1) / (y2 - y1))
    return round(float(target_frame), 2), float(target)


def interpolate_target_frame_nearest(xs, ys, lo: int, hi: int, target: float, ref_x: float):
    seg_x = np.array(xs[lo:hi + 1], dtype=float)
    seg_y = np.array(ys[lo:hi + 1], dtype=float)
    if len(seg_x) == 0:
        raise SystemExit("No samples available for interpolation.")

    candidates = []
    exact_hits = np.where(seg_y == target)[0]
    for idx in exact_hits:
        candidates.append(float(seg_x[int(idx)]))

    for i in range(len(seg_y) - 1):
        y1 = seg_y[i] - target
        y2 = seg_y[i + 1] - target
        if y1 * y2 < 0:
            x1 = seg_x[i]
            x2 = seg_x[i + 1]
            raw_y1 = seg_y[i]
            raw_y2 = seg_y[i + 1]
            if x1 == x2 or raw_y1 == raw_y2:
                frame = (x1 + x2) / 2.0
            else:
                frame = x1 + ((target - raw_y1) * (x2 - x1) / (raw_y2 - raw_y1))
            candidates.append(float(frame))

    if candidates:
        target_frame = min(candidates, key=lambda x: abs(x - ref_x))
        return round(float(target_frame), 2), float(target)

    return interpolate_target_frame(xs, ys, lo, hi, target, crossing_mode="best")


def interpolate_last_target_before(xs, ys, hi_idx: int, target: float, upper_x: float):
    seg_x = np.array(xs[:hi_idx + 1], dtype=float)
    seg_y = np.array(ys[:hi_idx + 1], dtype=float)
    if len(seg_x) == 0:
        raise SystemExit("No samples available for takeoff-point interpolation.")

    candidate_frame = None
    for i in range(len(seg_y) - 1):
        x1 = seg_x[i]
        x2 = seg_x[i + 1]
        y1 = seg_y[i]
        y2 = seg_y[i + 1]

        if y1 == target and x1 < upper_x:
            candidate_frame = x1
        if y2 == target and x2 < upper_x:
            candidate_frame = x2

        if (y1 - target) * (y2 - target) < 0:
            if x1 == x2 or y1 == y2:
                frame = (x1 + x2) / 2.0
            else:
                frame = x1 + ((target - y1) * (x2 - x1) / (y2 - y1))
            if frame < upper_x:
                candidate_frame = frame

    if candidate_frame is not None:
        return round(float(candidate_frame), 2)

    valid_idx = [i for i, x in enumerate(seg_x) if x < upper_x]
    if not valid_idx:
        raise SystemExit("No hip-y samples found before min-velocity frame.")

    ranked = sorted(
        valid_idx,
        key=lambda i: (abs(seg_y[i] - target), upper_x - seg_x[i]),
    )
    i1 = ranked[0]
    neighbor_candidates = []
    if i1 > 0:
        neighbor_candidates.append(i1 - 1)
    if i1 + 1 < len(seg_x):
        neighbor_candidates.append(i1 + 1)

    if not neighbor_candidates:
        return round(float(seg_x[i1]), 2)

    i2 = min(
        neighbor_candidates,
        key=lambda j: (max(abs(seg_y[i1] - target), abs(seg_y[j] - target)), abs(upper_x - seg_x[j])),
    )
    x1 = seg_x[i1]
    x2 = seg_x[i2]
    y1 = seg_y[i1]
    y2 = seg_y[i2]
    if x1 == x2 or y1 == y2:
        frame = min(max(x1, x2), upper_x)
    else:
        frame = x1 + ((target - y1) * (x2 - x1) / (y2 - y1))
    lo_x = min(x1, x2)
    hi_x = min(max(x1, x2), upper_x)
    frame = min(max(frame, lo_x), hi_x)
    if frame >= upper_x:
        frame = max(lo_x, upper_x - 0.01)
    return round(float(frame), 2)


def interpolate_series_value(xs, ys, x):
    if len(xs) == 0:
        raise SystemExit("No samples available for value interpolation.")
    xp = np.array(xs, dtype=float)
    yp = np.array(ys, dtype=float)
    return float(np.interp(float(x), xp, yp))


def smooth_dense_series(values, window: int):
    if not values:
        return []
    return smooth_valid_series(values, [True] * len(values), window)


def fit_local_quadratic_apex(xs, ys, center_frame: float, half_window: int = 2):
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    if len(x_arr) < 3:
        center_idx = int(np.argmin(y_arr))
        return round(float(x_arr[center_idx]), 2), float(y_arr[center_idx]), "min_height_fallback"

    center_idx = int(np.argmin(np.abs(x_arr - float(center_frame))))
    lo = max(0, center_idx - half_window)
    hi = min(len(x_arr), center_idx + half_window + 1)
    if hi - lo < 3:
        if lo == 0:
            hi = min(len(x_arr), 3)
        else:
            lo = max(0, len(x_arr) - 3)

    local_x = x_arr[lo:hi]
    local_y = y_arr[lo:hi]
    fallback_idx = int(np.argmin(local_y))
    fallback_x = float(local_x[fallback_idx])
    fallback_y = float(local_y[fallback_idx])

    if len(local_x) < 3 or len(np.unique(local_x)) < 3:
        return round(fallback_x, 2), fallback_y, "min_height_fallback"

    try:
        a, b, c = np.polyfit(local_x, local_y, 2)
    except np.linalg.LinAlgError:
        return round(fallback_x, 2), fallback_y, "min_height_fallback"

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c) or a <= 0.0:
        return round(fallback_x, 2), fallback_y, "min_height_fallback"

    apex_x = float(-b / (2.0 * a))
    local_lo = float(local_x[0])
    local_hi = float(local_x[-1])
    if apex_x < local_lo or apex_x > local_hi:
        return round(fallback_x, 2), fallback_y, "min_height_fallback"

    apex_y = float(np.polyval([a, b, c], apex_x))
    return round(apex_x, 2), apex_y, "quadratic_apex_fit"


def compute_jump_metrics(
    frames,
    fps: float,
    gravity: float,
    velocity_smooth_window: int = 1,
):
    xs = []
    ys = []
    for fr in frames:
        y = get_hip_y(fr)
        if y is None:
            continue
        xs.append(fr.get("frame_index", len(xs)))
        ys.append(y)

    if len(xs) < 2:
        raise SystemExit("Not enough valid hip keypoints to compute velocity.")

    # Downward velocity in image coordinates: dy = y[t] - y[t-1]
    raw_vxs = []
    raw_vys = []
    for i in range(1, len(xs)):
        raw_vxs.append(xs[i])
        raw_vys.append(ys[i] - ys[i - 1])

    vxs = list(raw_vxs)
    vys = smooth_dense_series(raw_vys, velocity_smooth_window)

    min_i = int(np.argmin(vys))
    min_x = float(vxs[min_i])
    max_i = int(np.argmax(vys))
    max_x = float(vxs[max_i])
    # Highest point: fit a local quadratic around the coarse apex region.
    highest_start = min_i
    highest_end = len(vxs) - 1
    velocity_zero_frame, _ = interpolate_target_frame(
        vxs, vys, highest_start, highest_end, 0.0, crossing_mode="first"
    )
    min_height_i = int(np.argmin(ys))
    min_height_frame = float(xs[min_height_i])
    coarse_highest_frame = (velocity_zero_frame + min_height_frame) / 2.0
    highest_point_frame, highest_point_y, highest_fit_method = fit_local_quadratic_apex(
        xs, ys, coarse_highest_frame
    )
    highest_point_velocity = interpolate_series_value(vxs, vys, highest_point_frame)

    if len(vxs) < 3:
        raise SystemExit("Not enough velocity samples to compute takeoff/landing points from acceleration.")
    axs = vxs[1:]
    ays = [vys[i] - vys[i - 1] for i in range(1, len(vys))]

    # Takeoff point: before the highest point, use the last acceleration zero-crossing.
    takeoff_end = None
    for i in range(len(axs) - 1, -1, -1):
        if axs[i] <= highest_point_frame:
            takeoff_end = i
            break
    if takeoff_end is None:
        raise SystemExit("No acceleration samples found before highest point.")
    takeoff_point_frame, takeoff_point_acceleration = interpolate_target_frame(
        axs, ays, 0, takeoff_end, 0.0, crossing_mode="last"
    )
    if takeoff_point_frame >= highest_point_frame:
        raise SystemExit("Invalid event order: takeoff point must be before highest point.")
    takeoff_point_velocity = interpolate_series_value(vxs, vys, takeoff_point_frame)

    # Landing point: after the highest point, use the acceleration zero-crossing nearest
    # to the max-velocity frame.
    landing_start = None
    for i, ax in enumerate(axs):
        if ax >= highest_point_frame:
            landing_start = i
            break
    if landing_start is None:
        raise SystemExit("No acceleration samples found after highest point.")
    landing_point_frame, landing_point_acceleration = interpolate_target_frame_nearest(
        axs, ays, landing_start, len(axs) - 1, 0.0, max_x
    )
    if landing_point_frame <= highest_point_frame:
        raise SystemExit("Invalid event order: landing point must be after highest point.")
    landing_point_velocity = interpolate_series_value(vxs, vys, landing_point_frame)

    if gravity == 0:
        raise SystemExit("gravity must not be 0.")
    t_frames_base = landing_point_frame - highest_point_frame
    frame_adjust = 0.0
    t_frames = t_frames_base + frame_adjust
    if t_frames <= 0:
        raise SystemExit("Invalid t frames: landing_point_frame - highest_point_frame must be positive.")
    flight_time_sec = t_frames / fps
    jump_height_m = 0.5 * gravity * (flight_time_sec ** 2)
    jump_height_cm = jump_height_m * 100.0

    return {
        "xs": xs,
        "ys": ys,
        "raw_vxs": raw_vxs,
        "raw_vys": raw_vys,
        "vxs": vxs,
        "vys": vys,
        "axs": axs,
        "ays": ays,
        "velocity_smooth_window": velocity_smooth_window,
        "takeoff_x": takeoff_point_frame,
        "takeoff_y": takeoff_point_velocity,
        "takeoff_acceleration": takeoff_point_acceleration,
        "highest_x": highest_point_frame,
        "highest_y": highest_point_velocity,
        "highest_position_y": highest_point_y,
        "highest_fit_method": highest_fit_method,
        "coarse_highest_frame": coarse_highest_frame,
        "velocity_zero_frame": velocity_zero_frame,
        "min_height_frame": min_height_frame,
        "landing_x": landing_point_frame,
        "landing_y": landing_point_velocity,
        "landing_acceleration": landing_point_acceleration,
        "takeoff_point_frame": takeoff_point_frame,
        "highest_point_frame": highest_point_frame,
        "landing_point_frame": landing_point_frame,
        "frame_adjust": frame_adjust,
        "t_frames": t_frames,
        "flight_time_sec": flight_time_sec,
        "jump_height_m": jump_height_m,
        "jump_height_cm": jump_height_cm,
    }


def parse_actual_heights_cm(text: str):
    vals = []
    for token in text.split(","):
        s = token.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("No valid heights found in --actual-heights-cm.")
    return vals


def parse_clip_index(text: str):
    s = str(text).strip().lower()
    m = re.fullmatch(r"clip[_-]?(\d+)", s)
    if m:
        return int(m.group(1))
    try:
        return int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid clip index '{text}'. Use values like 0 or clip_0."
        ) from exc


def normalize_subject_key(subject: str | None):
    if subject is None:
        return None
    return re.sub(r"[^a-z0-9]+", "", subject.lower())


def resolve_actual_heights_cm(text: str | None, subject: str | None):
    if text is not None:
        return parse_actual_heights_cm(text)

    subject_key = normalize_subject_key(subject)
    if subject_key is None:
        raise SystemExit(
            "Batch mode requires --actual-heights-cm when subject cannot be inferred."
        )
    if subject_key not in STORED_ACTUAL_HEIGHTS_CM:
        raise SystemExit(
            f"No stored actual heights for subject '{subject}'. "
            "Use --actual-heights-cm to provide them explicitly."
        )
    return STORED_ACTUAL_HEIGHTS_CM[subject_key]


def compute_jump_from_manual_frames(highest_point_frame: float, landing_point_frame: float, fps: float, gravity: float):
    if gravity == 0:
        raise SystemExit("gravity must not be 0.")
    t_frames = float(landing_point_frame) - float(highest_point_frame)
    if t_frames <= 0:
        raise SystemExit("Invalid manual frames: landing_point_frame - highest_point_frame must be positive.")
    flight_time_sec = t_frames / fps
    jump_height_m = 0.5 * gravity * (flight_time_sec ** 2)
    jump_height_cm = jump_height_m * 100.0
    return {
        "t_frames": t_frames,
        "flight_time_sec": flight_time_sec,
        "jump_height_m": jump_height_m,
        "jump_height_cm": jump_height_cm,
    }


def run_manual_error_mode(args):
    subject = args.subject_id
    if subject is None and args.batch_dir is not None:
        subject, _ = infer_subject_and_angle(args.batch_dir)
    if subject is None and args.input is not None:
        subject, _ = infer_subject_and_angle(args.input)
    if subject is None:
        raise SystemExit("Manual error mode requires --subject-id, or a path that includes the subject.")

    actual_heights = resolve_actual_heights_cm(args.actual_heights_cm, subject)
    clip_idx = args.manual_clip_index
    if clip_idx < 0 or clip_idx >= len(actual_heights):
        raise SystemExit(
            f"Invalid --manual-clip-index {clip_idx}. "
            f"Available range for {subject}: 0-{len(actual_heights) - 1}."
        )

    stats = compute_jump_from_manual_frames(
        args.manual_highest_point_frame,
        args.manual_landing_point_frame,
        args.fps,
        args.gravity,
    )
    actual_cm = actual_heights[clip_idx]
    error_rate = (actual_cm - stats["jump_height_cm"]) / actual_cm
    print(f"manual_error_rate,{error_rate:.6f}")
    print(f"manual_error_rate_percent,{error_rate*100:.3f}")


def infer_subject_and_angle(batch_dir: Path):
    subject = None
    angle = None
    parts = batch_dir.resolve().parts
    for i, part in enumerate(parts):
        if re.fullmatch(r"jumper[_-]?\d+", part):
            subject = part
            if i > 0:
                angle = parts[i - 1]
            break
    return subject, angle


def resolve_measurement_csv_path(args, subject: str, angle: str):
    filename = f"{subject}.csv" if subject != "unknown" else "measurement.csv"
    if args.csv_output is not None:
        return args.csv_output
    if args.measurement_dir is not None:
        return args.measurement_dir / filename
    if subject != "unknown" and angle != "unknown":
        return Path("/home/yirui/jump_video") / angle / subject / "measurement" / filename
    return args.batch_dir / "measurement" / filename


def draw_bottom_right_text(frame, text: str):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]
    margin = 0
    max_text_width = max(1, int(frame_w * 0.88))
    target_text_height = max(1, int(frame_h * 0.080))

    (base_w, base_h), _ = cv2.getTextSize(text, font, 1.0, 2)
    width_scale = max_text_width / max(base_w, 1)
    height_scale = target_text_height / max(base_h, 1)
    font_scale = min(max(2.8, height_scale), width_scale)
    thickness = max(3, int(round(font_scale * 2.2)))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, frame.shape[1] - text_w)
    y = max(text_h, frame.shape[0] - baseline)

    pad_x = max(10, int(round(min(frame_w, frame_h) * 0.012)))
    pad_y = max(8, int(round(min(frame_w, frame_h) * 0.010)))
    box_tl = (max(0, x - pad_x), max(0, y - text_h - pad_y))
    box_br = (frame.shape[1], frame.shape[0])
    cv2.rectangle(frame, box_tl, box_br, (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    frames,
    metrics,
    conf: float,
    radius: int,
    thickness: int,
):
    import cv2

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    landing_point_frame = float(metrics["landing_point_frame"])
    height_text = f"Vertical Jump Height: {metrics['jump_height_cm']:.3f} cm"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < len(frames):
            frame_data = frames[frame_idx]
            people = frame_data.get("people", [])
            if people:
                kpts = people[0].get("keypoints_xyc", [])

                for a, b in COCO_EDGES:
                    if a < len(kpts) and b < len(kpts):
                        xa, ya, ca = kpts[a]
                        xb, yb, cb = kpts[b]
                        if ca >= conf and cb >= conf:
                            cv2.line(
                                frame,
                                (int(xa), int(ya)),
                                (int(xb), int(yb)),
                                (0, 255, 0),
                                thickness,
                            )

                for x, y, c in kpts:
                    if c >= conf:
                        cv2.circle(frame, (int(x), int(y)), radius, (0, 0, 255), -1)

            current_frame_index = int(frame_data.get("frame_index", frame_idx))
            if current_frame_index >= landing_point_frame:
                draw_bottom_right_text(frame, height_text)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def plot_velocity(metrics, output: Path, title: str | None):
    vxs = metrics["vxs"]
    vys = metrics["vys"]
    velocity_smooth_window = metrics.get("velocity_smooth_window", 1)
    takeoff_x = metrics["takeoff_x"]
    takeoff_y = metrics["takeoff_y"]
    takeoff_acceleration = metrics["takeoff_acceleration"]
    highest_x = metrics["highest_x"]
    highest_y = metrics["highest_y"]
    landing_x = metrics["landing_x"]
    landing_y = metrics["landing_y"]
    landing_acceleration = metrics["landing_acceleration"]
    takeoff_point_frame = metrics["takeoff_point_frame"]
    highest_point_frame = metrics["highest_point_frame"]
    landing_point_frame = metrics["landing_point_frame"]
    frame_adjust = metrics["frame_adjust"]
    t_frames = metrics["t_frames"]
    flight_time_sec = metrics["flight_time_sec"]
    jump_height_m = metrics["jump_height_m"]
    jump_height_cm = metrics["jump_height_cm"]
    vel_label = "Left Hip Y Velocity" if velocity_smooth_window <= 1 else "Left Hip Y Velocity (smoothed)"

    plt.figure(figsize=(12, 4))
    plt.plot(vxs, vys, linewidth=1.0, label=vel_label)
    plt.scatter([takeoff_x], [takeoff_y], color="blue", zorder=3)
    plt.scatter([highest_x], [highest_y], color="orange", zorder=3)
    plt.scatter([landing_x], [landing_y], color="red", zorder=3)
    plt.annotate(f"landing(acc=0) @ frame {landing_x:.2f}", (landing_x, landing_y),
                 textcoords="offset points", xytext=(8, -12), color="red")
    plt.annotate(f"takeoff(acc=0) @ frame {takeoff_x:.2f}", (takeoff_x, takeoff_y),
                 textcoords="offset points", xytext=(8, 8), color="blue")
    plt.annotate(f"highest(apex fit) @ frame {highest_x:.2f}", (highest_x, highest_y),
                 textcoords="offset points", xytext=(8, -12), color="orange")
    plt.text(
        0.01,
        0.98,
        (
            f"takeoff={takeoff_point_frame:.2f}, highest={highest_point_frame:.2f}, landing={landing_point_frame:.2f}\n"
            f"t=landing-highest{frame_adjust:+.2f}={t_frames:.2f} frames\n"
            f"flight={flight_time_sec:.4f} s\n"
            f"jump height={jump_height_m:.4f} m ({jump_height_cm:.2f} cm)"
        ),
        transform=plt.gca().transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    plt.xlabel("Frame")
    plt.ylabel("Left Hip Y Velocity (pixel/frame, + = downward)")
    if title:
        plt.title(title)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.legend(loc="upper right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def plot_velocity_smoothing_comparison(raw_metrics, smooth_metrics, output: Path, title: str | None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    panels = [
        ("No Velocity Smoothing", raw_metrics, "#1f77b4"),
        (
            f"Velocity Smoothing (window={smooth_metrics.get('velocity_smooth_window', 1)})",
            smooth_metrics,
            "#d95f02",
        ),
    ]

    for ax, (panel_title, metrics, line_color) in zip(axes, panels):
        vxs = metrics["vxs"]
        vys = metrics["vys"]
        takeoff_x = metrics["takeoff_x"]
        takeoff_y = metrics["takeoff_y"]
        highest_x = metrics["highest_x"]
        highest_y = metrics["highest_y"]
        landing_x = metrics["landing_x"]
        landing_y = metrics["landing_y"]

        ax.plot(vxs, vys, color=line_color, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.7)
        ax.scatter([takeoff_x], [takeoff_y], color="#1f77b4", zorder=3)
        ax.scatter([highest_x], [highest_y], color="#ff7f0e", zorder=3)
        ax.scatter([landing_x], [landing_y], color="#d62728", zorder=3)
        ax.annotate(f"takeoff {takeoff_x:.2f}", (takeoff_x, takeoff_y), textcoords="offset points", xytext=(8, 8), color="#1f77b4")
        ax.annotate(f"highest {highest_x:.2f}", (highest_x, highest_y), textcoords="offset points", xytext=(8, -12), color="#ff7f0e")
        ax.annotate(f"landing {landing_x:.2f}", (landing_x, landing_y), textcoords="offset points", xytext=(8, -12), color="#d62728")
        ax.text(
            0.01,
            0.98,
            (
                f"flight={metrics['flight_time_sec']:.4f} s\n"
                f"height={metrics['jump_height_cm']:.2f} cm"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_ylabel("Velocity")
        ax.set_title(panel_title)

    axes[-1].set_xlabel("Frame")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pva_overlay(metrics, output: Path, title: str | None):
    xs = metrics["xs"]
    ys = metrics["ys"]
    vxs = metrics["vxs"]
    vys = metrics["vys"]
    axs = metrics["axs"]
    ays = metrics["ays"]
    velocity_smooth_window = metrics.get("velocity_smooth_window", 1)
    takeoff_point_frame = metrics["takeoff_point_frame"]
    highest_point_frame = metrics["highest_point_frame"]
    landing_point_frame = metrics["landing_point_frame"]

    fig, ax_pos = plt.subplots(figsize=(12, 5))
    ax_vel = ax_pos.twinx()
    ax_acc = ax_pos.twinx()
    ax_acc.spines["right"].set_position(("axes", 1.10))

    pos_line = ax_pos.plot(xs, ys, color="#1f77b4", linewidth=1.6, label="Left Hip Y Position")[0]
    vel_label = "Left Hip Y Velocity" if velocity_smooth_window <= 1 else "Left Hip Y Velocity (smoothed)"
    acc_label = "Left Hip Y Acceleration"
    vel_line = ax_vel.plot(vxs, vys, color="#ff7f0e", linewidth=1.4, label=vel_label)[0]
    acc_line = ax_acc.plot(axs, ays, color="#2ca02c", linewidth=1.2, label=acc_label)[0]

    for x, color, label in (
        (takeoff_point_frame, "#1f77b4", "takeoff"),
        (highest_point_frame, "#ff7f0e", "highest"),
        (landing_point_frame, "#d62728", "landing"),
    ):
        ax_pos.axvline(x, color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    ax_pos.set_xlabel("Frame")
    ax_pos.set_ylabel("Left Hip Y Position (pixel)", color=pos_line.get_color())
    ax_vel.set_ylabel("Left Hip Y Velocity (pixel/frame)", color=vel_line.get_color())
    ax_acc.set_ylabel("Left Hip Y Acceleration (pixel/frame^2)", color=acc_line.get_color())
    ax_pos.tick_params(axis="y", colors=pos_line.get_color())
    ax_vel.tick_params(axis="y", colors=vel_line.get_color())
    ax_acc.tick_params(axis="y", colors=acc_line.get_color())
    ax_vel.axhline(0, color=vel_line.get_color(), linewidth=0.8, alpha=0.4)
    ax_acc.axhline(0, color=acc_line.get_color(), linewidth=0.8, alpha=0.4)

    lines = [pos_line, vel_line, acc_line]
    labels = [line.get_label() for line in lines]
    ax_pos.legend(lines, labels, loc="upper left")
    event_box = dict(facecolor="white", alpha=0.8, edgecolor="none")
    ax_pos.text(
        0.015,
        0.16,
        f"takeoff: {takeoff_point_frame:.2f}",
        transform=ax_pos.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#1f77b4",
        bbox=event_box,
    )
    ax_pos.text(
        0.015,
        0.09,
        f"highest: {highest_point_frame:.2f}",
        transform=ax_pos.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#ff7f0e",
        bbox=event_box,
    )
    ax_pos.text(
        0.015,
        0.02,
        f"landing: {landing_point_frame:.2f}",
        transform=ax_pos.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#d62728",
        bbox=event_box,
    )
    if title:
        ax_pos.set_title(f"{title} PVA overlay")
    else:
        ax_pos.set_title("Left Hip Y Position / Velocity / Acceleration Overlay")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_batch_mode(args):
    inferred_subject, inferred_angle = infer_subject_and_angle(args.batch_dir)
    subject = args.subject_id if args.subject_id is not None else inferred_subject
    angle = args.camera_angle if args.camera_angle is not None else inferred_angle
    if subject is None:
        subject = "unknown"
    if angle is None:
        angle = "unknown"
    actual_heights = resolve_actual_heights_cm(args.actual_heights_cm, subject)

    clip_files = []
    for p in args.batch_dir.glob("clip_*_keypoints2d.json"):
        m = re.fullmatch(r"clip_(\d+)_keypoints2d\.json", p.name)
        if not m:
            continue
        clip_idx = int(m.group(1))
        clip_files.append((clip_idx, p))

    if not clip_files:
        raise SystemExit(f"No clip_*_keypoints2d.json found in {args.batch_dir}")

    clip_files.sort(key=lambda t: t[0])

    clip_ids = []
    error_rates = []
    csv_rows = []

    for clip_idx, p in clip_files:
        if clip_idx >= len(actual_heights):
            raise SystemExit(
                f"Missing actual height for clip_{clip_idx}. "
                f"Provide at least {clip_idx + 1} values in --actual-heights-cm."
            )
        frames, metrics = prepare_frames_and_metrics(
            load_frames(p),
            args.smooth,
            args.fps,
            args.gravity,
            args.velocity_smooth,
        )
        measured_cm = metrics["jump_height_cm"]
        actual_cm = actual_heights[clip_idx]
        if actual_cm == 0:
            raise SystemExit(f"Actual height for clip_{clip_idx} must not be 0.")
        actual_height_m = actual_cm / 100.0
        expected_flight_time_sec = np.sqrt((2.0 * actual_height_m) / args.gravity)
        expected_flight_time_frames = expected_flight_time_sec * args.fps
        expected_minus_flight_time_frames = expected_flight_time_frames - metrics["t_frames"]
        err = (actual_cm - measured_cm) / actual_cm

        clip_ids.append(clip_idx)
        error_rates.append(err)
        csv_rows.append(
            {
                "subject": subject,
                "angle": angle,
                "clip": f"clip_{clip_idx}",
                "actual_height_cm": f"{actual_cm:.3f}",
                "measured_height_cm": f"{measured_cm:.3f}",
                "error_rate": f"{err:.6f}",
                "takeoff_point_frame": f"{metrics['takeoff_point_frame']:.2f}",
                "highest_point_frame": f"{metrics['highest_point_frame']:.2f}",
                "landing_point_frame": f"{metrics['landing_point_frame']:.2f}",
                "expected_flight_time_frames": f"{expected_flight_time_frames:.2f}",
                "flight_time_frames": f"{metrics['t_frames']:.2f}",
                "expected_minus_flight_time_frames": f"{expected_minus_flight_time_frames:.2f}",
                "flight_time_sec": f"{metrics['flight_time_sec']:.3f}",
            }
        )
    error_arr = np.array(error_rates, dtype=float)
    mean_abs_error_ratio = float(np.mean(np.abs(error_arr)))
    print(f"mean_absolute_error_rate_percent,{mean_abs_error_ratio*100:.3f}")
    mean_abs_error_percent = mean_abs_error_ratio * 100.0
    for row in csv_rows:
        row["mean_absolute_error_rate_percent"] = f"{mean_abs_error_percent:.3f}"

    csv_output = resolve_measurement_csv_path(args, subject, angle)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "angle",
                "clip",
                "actual_height_cm",
                "measured_height_cm",
                "error_rate",
                "takeoff_point_frame",
                "highest_point_frame",
                "landing_point_frame",
                "expected_flight_time_frames",
                "flight_time_frames",
                "expected_minus_flight_time_frames",
                "flight_time_sec",
                "mean_absolute_error_rate_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"measurement_csv: {csv_output}")
    if args.skip_error_plot:
        return

    error_output = args.error_output
    if error_output is None:
        error_output = args.batch_dir / "clip_error_rate.png"

    x = np.array(clip_ids, dtype=int)
    y = np.array(error_rates, dtype=float) * 100.0
    colors = ["#2a9d8f" if v >= 0 else "#d1495b" for v in y]

    plt.figure(figsize=(12, 4))
    plt.bar(x, y, color=colors, alpha=0.9)
    plt.axhline(0, color="gray", linewidth=0.8)
    for xi, yi in zip(x, y):
        va = "bottom" if yi >= 0 else "top"
        yoff = 0.35 if yi >= 0 else -0.35
        plt.text(xi, yi + yoff, f"{yi:.2f}%", ha="center", va=va, fontsize=8)
    plt.xticks(x, [f"clip{idx}" for idx in clip_ids])
    plt.ylabel("Error Rate (%) = (actual - measured) / actual * 100")
    plt.xlabel("Clip")
    if args.title:
        plt.title(f"{args.title} error rate")
    else:
        plt.title(f"{args.batch_dir.name} error rate")
    plt.tight_layout()
    error_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(error_output, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot hip Y position/velocity/acceleration and estimate jump "
            "flight time/height from hip motion."
        )
    )
    ap.add_argument("--input", type=Path, default=None, help="Path to *_keypoints2d.json. Optional when --video is used for direct YOLO detection")
    ap.add_argument("--video", type=Path, default=None, help="Input video path for overlay output or direct YOLO detection")
    ap.add_argument("--output", type=Path, default=None, help="Path to output image (png) or overlay video (mp4 when --video is used)")
    ap.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Path to output a two-panel comparison image with and without velocity smoothing.",
    )
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    ap.add_argument("--conf", type=float, default=0.1, help="Min keypoint confidence for video overlay")
    ap.add_argument("--radius", type=int, default=3, help="Keypoint circle radius for video overlay")
    ap.add_argument("--thickness", type=int, default=2, help="Skeleton line thickness for video overlay")
    ap.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("/home/yirui/mydata/yolo11x/yolo11x-pose.pt"),
        help="YOLO11x pose model path for direct video detection",
    )
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO pose confidence threshold for direct video detection")
    ap.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO pose IOU threshold for direct video detection")
    ap.add_argument("--yolo-device", type=str, default="cpu", help="YOLO device for direct video detection, e.g. cpu or cuda:0")
    ap.add_argument(
        "--smooth",
        type=int,
        default=9,
        help="Temporal smoothing window for keypoint coordinates",
    )
    ap.add_argument(
        "--velocity-smooth",
        type=int,
        default=9,
        help="Temporal smoothing window for the derived velocity waveform before jump-height calculation.",
    )
    ap.add_argument("--fps", type=float, default=60.0, help="Video FPS for flight-time calculation")
    ap.add_argument("--gravity", type=float, default=9.81, help="Gravity (m/s^2) for jump-height calculation")
    ap.add_argument("--batch-dir", type=Path, default=None, help="Directory containing clip_*_keypoints2d.json")
    ap.add_argument(
        "--actual-heights-cm",
        type=str,
        default=None,
        help="Comma-separated actual heights in cm. Index must match clip number. Optional if subject has stored heights.",
    )
    ap.add_argument(
        "--error-output",
        type=Path,
        default=None,
        help="Output image path for batch error-rate plot (png).",
    )
    ap.add_argument(
        "--skip-error-plot",
        action="store_true",
        help="Skip writing the batch error-rate plot and only export CSV.",
    )
    ap.add_argument(
        "--measurement-dir",
        type=Path,
        default=None,
        help="Directory to save measurement.csv in batch mode.",
    )
    ap.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Explicit CSV output path in batch mode.",
    )
    ap.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject id for CSV (e.g., jumper_1). Auto-inferred from --batch-dir if omitted.",
    )
    ap.add_argument(
        "--camera-angle",
        type=str,
        default=None,
        help="Camera angle for CSV (e.g., FL). Auto-inferred from --batch-dir if omitted.",
    )
    ap.add_argument(
        "--manual-clip-index",
        "--manual-clip",
        dest="manual_clip_index",
        type=parse_clip_index,
        default=None,
        help="Manual error mode clip index, e.g. 0 or clip_0.",
    )
    ap.add_argument(
        "--manual-highest-point-frame",
        type=float,
        default=None,
        help="Manual error mode highest-point frame.",
    )
    ap.add_argument(
        "--manual-landing-point-frame",
        type=float,
        default=None,
        help="Manual error mode landing-point frame.",
    )
    args = ap.parse_args()

    manual_mode_requested = any(
        value is not None
        for value in (
            args.manual_clip_index,
            args.manual_highest_point_frame,
            args.manual_landing_point_frame,
        )
    )
    if manual_mode_requested:
        missing = []
        if args.manual_clip_index is None:
            missing.append("--manual-clip-index")
        if args.manual_highest_point_frame is None:
            missing.append("--manual-highest-point-frame")
        if args.manual_landing_point_frame is None:
            missing.append("--manual-landing-point-frame")
        if missing:
            raise SystemExit(f"Manual error mode is missing required arguments: {', '.join(missing)}")
        run_manual_error_mode(args)
        return

    if args.batch_dir is not None:
        run_batch_mode(args)
        return

    if (args.output is None and args.comparison_output is None) or (args.input is None and args.video is None):
        raise SystemExit("Single mode requires --output or --comparison-output and either --input or --video.")

    if args.input is not None:
        frames = load_frames(args.input)
    else:
        frames = detect_pose_frames(
            args.video,
            args.yolo_model,
            args.yolo_conf,
            args.yolo_iou,
            args.yolo_device,
        )

    if args.comparison_output is not None:
        _, raw_metrics = prepare_frames_and_metrics(
            frames,
            args.smooth,
            args.fps,
            args.gravity,
            1,
        )
        _, smooth_metrics = prepare_frames_and_metrics(
            frames,
            args.smooth,
            args.fps,
            args.gravity,
            args.velocity_smooth,
        )
        plot_velocity_smoothing_comparison(raw_metrics, smooth_metrics, args.comparison_output, args.title)
        return

    frames, metrics = prepare_frames_and_metrics(
        frames,
        args.smooth,
        args.fps,
        args.gravity,
        args.velocity_smooth,
    )
    if args.video is not None:
        render_overlay_video(
            args.video,
            args.output,
            frames,
            metrics,
            args.conf,
            args.radius,
            args.thickness,
        )
    else:
        plot_pva_overlay(metrics, args.output, args.title)


if __name__ == "__main__":
    main()