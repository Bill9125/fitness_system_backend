import os
import torch
from pathlib import Path
from ultralytics import YOLO

def detect_jump_pose(video_path, model_path='fitness_system_models/deadlift/yolo11x-pose.pt', conf=0.25, iou=0.7, device=None):
    """
    Detect human pose keypoints from a video using YOLO.
    Handles any video resolution and automatically returns keypoints in original pixel coordinates.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    # stream=True is used to process long videos without consuming too much memory
    results = model.predict(
        source=video_path,
        stream=True,
        conf=conf,
        iou=iou,
        verbose=False,
        device=device
    )
    
    all_frames_data = []
    
    for frame_idx, result in enumerate(results):
        # We only care about the first detected person (usually the main jumper)
        # but the structure allows for multiple people if needed.
        frame_entry = {
            "frame_index": frame_idx,
            "people": []
        }
        
        if result.keypoints is not None:
            # keypoints.data has shape [num_people, num_keypoints, 3] (x, y, confidence)
            # Ultralytics already scales these back to the original video dimensions
            kpts_data = result.keypoints.data.cpu().numpy()
            
            for person_idx in range(kpts_data.shape[0]):
                frame_entry["people"].append({
                    "person_id": person_idx,
                    "keypoints_xyc": kpts_data[person_idx].tolist()
                })
        
        all_frames_data.append(frame_entry)
        
    return all_frames_data

if __name__ == "__main__":
    # Quick test usage
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python yolo_inference.py <video_path> <model_path>")
        sys.exit(1)
        
    v_path = sys.argv[1]
    m_path = sys.argv[2]
    
    print(f"Detecting poses in {v_path} using {m_path}...")
    output = detect_jump_pose(v_path, m_path)
    
    output_json = f"{Path(v_path).stem}_kpts.json"
    with open(output_json, 'w') as f:
        json.dump(output, f)
    
    print(f"Done! Saved {len(output)} frames to {output_json}")
