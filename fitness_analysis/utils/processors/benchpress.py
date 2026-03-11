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
    BENCHPRESS_TOP_SKELETON_CONNECTIONS,
    BENCHPRESS_REAR_SKELETON_CONNECTIONS,
)
from django.conf import settings
from ..tools.interpolate import run_interpolation
from ..tools.bar_data_produce import run_bar_data_produce
from ..tools.Benchpress_tool.hampel import (
    run_hampel_bar,
    run_hampel_yolo_ske_rear,
    run_hampel_yolo_ske_top
)
from ..tools.Benchpress_tool.torso_angle_produce import run_torso_angle_produce
from ..tools.Benchpress_tool.autocutting import run_autocutting
from ..tools.Benchpress_tool.predict import run_predict

class BenchpressProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.bar_model_path = settings.BAR_MODEL_PATH
        self.rear_pose_model_path = settings.BENCHPRESS_REAR_POSE_MODEL_PATH
        self.top_pose_model_path = settings.BENCHPRESS_TOP_POSE_MODEL_PATH
        self.rear_skeleton_connections = BENCHPRESS_REAR_SKELETON_CONNECTIONS
        self.top_skeleton_connections = BENCHPRESS_TOP_SKELETON_CONNECTIONS

    def load_models(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.models['bar'] = YOLO(self.bar_model_path).to(device)
        self.models['rear_pose'] = YOLO(self.rear_pose_model_path).to(device)
        self.models['top_pose'] = YOLO(self.top_pose_model_path).to(device)
        
        # Warmup
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        with torch.no_grad():
            self.models['bar'].predict([dummy], verbose=False)
            self.models['rear_pose'].predict([dummy], verbose=False)
            self.models['top_pose'].predict([dummy], verbose=False)
            
        print(f"[BenchpressProcessor] Models loaded on {device}")

    def process_frame(self, frame_data: dict):
        frame = frame_data['frame']
        idx = frame_data['cap_index']
        frame_cnt = frame_data['frame_count']
        bar_file = frame_data['bar_file']
        
        # Main View (Cam 0): Bar + Pose
        if idx == 0:
            # Note: bar_frame expects these specific args
            # We adapt our dictionary context to the function signature
            processed_frame, sk_list, b_list = bar_frame(
                frame,
                self.models['bar'],
                bar_file,
                frame_cnt
            )
            if bar_file and b_list:
                bar_file.writelines(b_list)
                
            return processed_frame, sk_list
            
        elif idx == 1:
            # Side Views: Pose Only
            processed_frame, sk_list = bone_frame(
                frame, 
                self.models['rear_pose'], 
                self.rear_skeleton_connections, 
                frame_cnt
            )
            return processed_frame, sk_list
        
        elif idx == 2:
            # Side Views: Pose Only
            processed_frame, sk_list = bone_frame(
                frame, 
                self.models['top_pose'], 
                self.top_skeleton_connections, 
                frame_cnt
            )
            return processed_frame, sk_list
        
        else:
            raise ValueError(f"Invalid idx: {idx}")

    def post_process(self, video_path: str):
        import time
        memo = {}
        
        def run_step(name, func, args, kwargs={}):
            t0 = time.time()
            res = func(*args, **kwargs)
            memo[name] = res
            print(f"[BenchpressProcessor] {name} time : {time.time() - t0:.2f}s")
            return res

        run_step("Interpolation", run_interpolation, [video_path])
        run_step("Bar Data", run_bar_data_produce, [video_path], {"sport": 'benchpress'})
        bar_dict = run_step("Hampel Bar", run_hampel_bar, [video_path])
        rear_ske_dict = run_step("Hampel Rear", run_hampel_yolo_ske_rear, [video_path])
        top_ske_dict = run_step("Hampel Top", run_hampel_yolo_ske_top, [video_path])
        run_step("Angle Data", run_torso_angle_produce, [video_path], {"skeleton_dict": top_ske_dict})
        split_info = run_step("Autocutting", run_autocutting, [video_path], {"bar_dict": bar_dict, "rear_ske_dict": rear_ske_dict})
        run_step("Predicting", run_predict, [video_path, bar_dict, rear_ske_dict, top_ske_dict, split_info])

    def get_result(self, folder: str, recording: 'Recording'):
        result = {}
        score_json_path = os.path.join(folder, "config/Score.json")
        bar_position_json_path = os.path.join(folder, "config/Bar_Position.json")
        split_info_json_path = os.path.join(folder, "config/Split_info.json")
        torso_angle_json_path = os.path.join(folder, "config/Torso_Angle.json")
        
        if not os.path.exists(score_json_path):
            return None
        
        with open(score_json_path, mode='r', encoding='utf-8') as json_file:
            score_data = json.load(json_file)
        with open(bar_position_json_path, mode='r', encoding='utf-8') as json_file:
            bar_position_data = json.load(json_file)
        with open(split_info_json_path, mode='r', encoding='utf-8') as json_file:
            split_info_data = json.load(json_file)
        with open(torso_angle_json_path, mode='r', encoding='utf-8') as json_file:
            torso_angle_data = json.load(json_file)
        
        # Write to DB only if a Recording ORM object is provided
        if recording is not None:
            import cv2
            bar_vid_path = os.path.join(folder, "vision_bar.mp4")
            if not os.path.exists(bar_vid_path):
                bar_vid_path = os.path.join(folder, "vision_bar.avi")
            
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
            'torso_angle': torso_angle_data,
            'split_info': split_info_data
        }
        return result
