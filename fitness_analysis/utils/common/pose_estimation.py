import time
import cv2
import numpy as np
import torch

def bar_frame(frame,
                bar_model,
                bar_file, 
                frame_count_for_detect,
                skeleton_connections=None,
                bone_model=None):
    """
    Process a single frame for bar detection and pose estimation (Lateral View).
    """
    # 1. Bar Detection
    results = bar_model(source=frame, imgsz=320, conf=0.5, verbose=False, device="cuda:0")
    boxes = results[0].boxes
    detected = False
    
    # Draw boxes
    for result in results:
        frame = result.plot()
    
    # Collect bar data
    bar_data_list = []
    
    for box in boxes.xywh:
        detected = True
        x_center, y_center, width, height = box.cpu().numpy() # Ensure cpu numpy
        # Format: frame, x, y, w, h
        bar_data_list.append(
            f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n"
        )

    if not detected:
        bar_data_list.append(f"{frame_count_for_detect},no detection\n")
        
    # 2. Pose Estimation (Bone Model)
    # Re-use common logic for bone frame
    if bone_model:
        frame, skeleton_data_list = _process_bone_frame(frame, bone_model, skeleton_connections, frame_count_for_detect)
    else:
        skeleton_data_list = []
    return frame, skeleton_data_list, bar_data_list

def bone_frame(frame,
                model,
                skeleton_connections,
                frame_count_for_detect):
    """
    Process a single frame for pose estimation only (Side Views).
    """
    return _process_bone_frame(frame, model, skeleton_connections, frame_count_for_detect)

def _process_bone_frame(frame, model, skeleton_connections, frame_count_for_detect):
    """
    Internal helper to run pose estimation and drawing.
    """
    results = list(model(source=frame, verbose=False, device="cuda:0"))
    skeleton_data_list = []
    
    if results and results[0].keypoints:
        keypoints = results[0].keypoints
        frame_h, frame_w = frame.shape[:2]
        center_frame = np.array([frame_w / 2, frame_h / 2])

        min_dist = float('inf')
        target_kpts = None

        # Find the person closest to center
        for kp in keypoints:
            coords = kp.xy[0].cpu().numpy()
            valid_coords = coords[(coords != 0).all(axis=1)]
            if len(valid_coords) == 0:
                continue
            person_center = np.mean(valid_coords, axis=0)
            dist = np.linalg.norm(person_center - center_frame)
            if dist < min_dist:
                min_dist = dist
                target_kpts = coords

        if target_kpts is not None:
            keypoints_xy = target_kpts
        else:
            keypoints_xy = []

        kp_coords = []
        if len(keypoints_xy) > 0:
            for idx, kp in enumerate(keypoints_xy):
                x_kp, y_kp = int(kp[0]), int(kp[1])

                if x_kp == 0 and y_kp == 0:
                    kp_coords.append(None)
                else:
                    kp_coords.append((x_kp, y_kp))
                    cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)

                skeleton_data_list.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}\n")

            # Draw connections
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                    if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
                        continue
                    cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx],
                            (0, 255, 255), 2)
        else:
             skeleton_data_list.append(f"{frame_count_for_detect},no detection\n")

    else:
        skeleton_data_list.append(f"{frame_count_for_detect},no detection\n")
        
    return frame, skeleton_data_list
