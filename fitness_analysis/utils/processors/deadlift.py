from django.conf import settings
import torch
import numpy as np
from ultralytics import YOLO

from ..base_processor import BaseProcessor
from ..common import (
    bar_frame, 
    bone_frame,
    SKELETON_CONNECTIONS,
)
from ..tools.Deadlift_tool.interpolate import run_interpolation
from ..tools.Deadlift_tool.bar_data_produce import run_bar_data_produce
from ..tools.Deadlift_tool.data_produce import run_data_produce
from ..tools.Deadlift_tool.data_split import run_data_split
from ..tools.Deadlift_tool.predict import run_predict
from ..tools.trajectory import plot_trajectory

class DeadliftProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.bar_model_path = settings.BAR_MODEL_PATH
        self.pose_model_path = settings.DEADLIFT_POSE_MODEL_PATH
        self.skeleton_connections = SKELETON_CONNECTIONS

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
                self.models['pose'], 
                self.skeleton_connections, 
                bar_file, 
                frame_cnt
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
        steps = [
            ("Interpolation", run_interpolation, [video_path]),
            ("Bar Data", lambda f: run_bar_data_produce(f, sport='deadlift'), [video_path]),
            ("Angle Data", run_data_produce, [video_path]),
            ("Data Split", run_data_split, [video_path]),
            ("Trajectory Plot", plot_trajectory, [video_path]),
            ("Prediction", run_predict, [video_path])
        ]
        
        import time
        for name, func, args in steps:
            t0 = time.time()
            func(*args)
            print(f"[DeadliftProcessor] {name} time :", time.time() - t0)
