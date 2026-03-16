from django.conf import settings
import torch
import numpy as np
from ultralytics import YOLO
import json
import os

from ..base_processor import BaseProcessor
from ..common import (
    bar_frame, 
    bone_frame,
    DEADLIFT_SKELETON_CONNECTIONS,
)
from ..tools.interpolate import run_interpolation
from ..tools.Benchpress_tool.hampel import run_hampel_bar, run_hampel_yolo_ske_left_front

from ..tools.Deadlift_tool.data_produce import run_data_produce
from ..tools.Deadlift_tool.data_split import run_data_split
from ..tools.Deadlift_tool.predict import run_predict
from ..tools.trajectory import plot_trajectory


class DeadliftProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.bar_model_path = settings.BAR_MODEL_PATH
        self.pose_model_path = settings.DEADLIFT_POSE_MODEL_PATH
        self.skeleton_connections = DEADLIFT_SKELETON_CONNECTIONS

    def load_models(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.models['bar'] = YOLO(self.bar_model_path).to(device)
        self.models['pose'] = YOLO(self.pose_model_path).to(device)
        
        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            self.models['bar'].predict([dummy], verbose=False)
            self.models['pose'].predict([dummy], verbose=False)
            
        print(f"[DeadliftProcessor] Models loaded on {device}")

    def process_frame(self, context: dict):
        frame = context['frame']
        idx = context['cap_index']
        frame_cnt = context['frame_count']
        bar_file = context['bar_file']
        
        # Main View (Cam 0): Bar + Pose
        if idx == 0:
            # Note: bar_frame expects these specific args
            # We adapt our dictionary context to the function signature
            processed_frame, sk_list, b_list = bar_frame(
                frame, 
                self.models['bar'], 
                bar_file, 
                frame_cnt,
                skeleton_connections=self.skeleton_connections,
                bone_model=self.models['pose'],
                draw=False
            )


            
            # For Deadlift, bar data is also stored in specific format in b_list
            # But BaseProcessor currently stores only ONE result object per frame per cam
            # This is a small mismatch. The BaseProcessor naively stores "frame_result".
            # We can merge sk_list and b_list strings, or handle bar writing inside bar_frame directly (which it seems to do partially?)
            # Re-checking bar_frame: it RETURNS bar_data list.
            
            # To fit BaseProcessor storage, we can append bar data to the "result list" 
            # OR we handle bar writing immediately here if we want to bypass storage.
            # But `_cleanup_resources` writes from storage.
            # Let's attach bar data as a special attribute? NO, `frame_data_storage` is simple.
            
            # HACK: The original code stored bar data in `bar_data` map and skeleton in `skeleton_data` map.
            # BaseProcessor merges everything into one list of lines? 
            # No, `BaseProcessor` has `skeleton_files[i].writelines(buffer)`.
            # Bar file is written separately.
            
            # SOLUTION: We should write to bar_file lines immediately or store them.
            # In `BaseProcessor`, `_cleanup_resources` only handles skeleton keys.
            # Let's do a trick: Write bar data directly here if we have the file handle?
            # Actually, `bar_frame` logic returns lines. We can just write them if file is open.
            
            if bar_file and b_list:
                bar_file.writelines(b_list)
                
            return processed_frame, sk_list
            
        else:
            # Side Views: Pose Only
            processed_frame, sk_list = bone_frame(
                frame, 
                self.models['pose'], 
                self.skeleton_connections, 
                frame_cnt
            )
            return processed_frame, sk_list

    def post_process(self, video_path: str):
        # Run the standard pipeline
        import time
        memo = {}
        
        def run_step(name, func, args, kwargs={}):
            t0 = time.time()
            res = func(*args, **kwargs)
            memo[name] = res
            print(f"[DeadliftProcessor] {name} time : {time.time() - t0:.2f}s")
            return res

        run_step("Interpolation", run_interpolation, [video_path])
        run_step("Hampel Bar", run_hampel_bar, [video_path], {"sport": 'deadlift'})
        run_step("Hampel Skeleton", run_hampel_yolo_ske_left_front, [video_path])
        run_step("Angle Data", run_data_produce, [video_path])
        run_step("Data Split", run_data_split, [video_path])
        run_step("Trajectory Plot", plot_trajectory, [video_path])
        run_step("Prediction", run_predict, [video_path])


            
    def get_result(self, video_path: str, recording=None):
        result = {}
        score_json_path = os.path.join(video_path, "config/Score.json")
        bar_position_json_path = os.path.join(video_path, "config/Bar_Position.json")
        hip_angle_json_path = os.path.join(video_path, "config/Hip_Angle.json")
        knee_angle_json_path = os.path.join(video_path, "config/Knee_Angle.json")
        knee_to_hip_json_path = os.path.join(video_path, "config/Knee_to_Hip.json")
        split_info_json_path = os.path.join(video_path, "config/Split_info.json")
        
        if not os.path.exists(score_json_path):
            return None
        
        with open(score_json_path, mode='r', encoding='utf-8') as json_file:
            score_data = json.load(json_file)['results']
        with open(bar_position_json_path, mode='r', encoding='utf-8') as json_file:
            bar_position_data = json.load(json_file)
        with open(hip_angle_json_path, mode='r', encoding='utf-8') as json_file:
            hip_angle_data = json.load(json_file)
        with open(knee_angle_json_path, mode='r', encoding='utf-8') as json_file:
            knee_angle_data = json.load(json_file)
        with open(knee_to_hip_json_path, mode='r', encoding='utf-8') as json_file:
            knee_to_hip_data = json.load(json_file)
        with open(split_info_json_path, mode='r', encoding='utf-8') as json_file:
            split_info_data = json.load(json_file)
        
        # Write to DB only if a Recording ORM object is provided
        if recording is not None:
            import cv2
            bar_vid_path = os.path.join(video_path, "vision_bar.mp4")
            if not os.path.exists(bar_vid_path):
                bar_vid_path = os.path.join(video_path, "vision_bar.avi")
            
            if os.path.exists(bar_vid_path):
                cap = cv2.VideoCapture(bar_vid_path)
                if cap.isOpened():
                    recording.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    recording.save()
                cap.release()

            from fitness_analysis.models import Repetition
            for key, val in split_info_data.items():
                rep_score = score_data.get(key, {}).get("score", 0)
                errors = ','.join(
                    error for error, conf in score_data.get(key, {}).items()
                    if error != "score" and conf >= 0.5
                )
                Repetition.objects.update_or_create(
                    recording=recording,
                    start_frame=val.get('start'),
                    defaults={
                        'end_frame': val.get('end'),
                        'score': rep_score,
                        'error': errors
                    }
                )

        result = {
            'score': score_data,
            'bar_position': bar_position_data,
            'hip_angle': hip_angle_data,
            'knee_angle': knee_angle_data,
            'knee_to_hip': knee_to_hip_data,
            'split_info': split_info_data
        }
        return result
