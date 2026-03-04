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
from ..tools.Benchpress_tool.torsor_angle_produce import run_torsor_angle_produce
from ..tools.Benchpress_tool.autocutting import run_autocutting
# from ..tools.Benchpress_tool.step5_calculate_angle_new_feature_test import run_calculate_angle_new_feature_test
# from ..tools.Benchpress_tool.step6_cut import run_cut
# from ..tools.Benchpress_tool.step7_length_100 import run_length_100

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
        run_step("Hampel Rear", run_hampel_yolo_ske_rear, [video_path])
        top_ske_dict = run_step("Hampel Top", run_hampel_yolo_ske_top, [video_path])
        
        run_step("Angle Data", run_torsor_angle_produce, [video_path], {"skeleton_dict": top_ske_dict})
        run_step("Autocutting", run_autocutting, [video_path], {"bar_dict": bar_dict, "top_ske_dict": top_ske_dict})
        
        # These are currently commented out in the original file, I'll keep them as placeholders if needed
        # run_step("Data Split", run_data_split, [video_path])
        # run_step("Trajectory Plot", plot_trajectory, [video_path])
        # run_step("Prediction", run_predict, [video_path])

# f'python ./tools/Benchpress_tool/offline_benchpress_head.py "{folder}"', 不需要
# f'python ./tools/Benchpress_tool/interpolate.py "{folder}"',
# f'python ./tools/Benchpress_tool/step0_hampel_bar.py "{folder}"',
# f'python ./tools/Benchpress_tool/bar_data_produce.py "{folder}" --out ./config --sport benchpress',
# f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_rear.py "{folder}"',
# f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_top.py "{folder}"',
# f'python ./tools/Benchpress_tool/step1_interpolate_bar.py "{folder}"',
# f'python ./tools/Benchpress_tool/step2_interpolate_yolo_ske.py "{folder}"',
# f'python ./tools/Benchpress_tool/torsor_angle_produce.py "{folder}"',
# f'python ./tools/Benchpress_tool/step3_autocutting_0801.py "{folder}"',
# f'python ./tools/Benchpress_tool/step5_calculate_angle_new_feature_test.py "{folder}"',
# f'python ./tools/Benchpress_tool/step6_cut.py "{folder}"', 不需要
# f'python ./tools/Benchpress_tool/step7_length_100.py "{folder}"',
# f'python ./tools/Benchpress_tool/step8_normalize.py "{folder}"',

    def get_result(self, folder: str, recording: 'Recording'):
        result = {}
        score_json_path = os.path.join(folder, "config/Score.json")
        bar_position_json_path = os.path.join(folder, "config/Bar_Position.json")
        split_info_json_path = os.path.join(folder, "config/Split_info.json")
        
        if not os.path.exists(score_json_path):
            return None
        
        with open(score_json_path, mode='r', encoding='utf-8') as json_file:
            score_data = json.load(json_file)['results']
        with open(bar_position_json_path, mode='r', encoding='utf-8') as json_file:
            bar_position_data = json.load(json_file)
        # with open(hip_angle_json_path, mode='r', encoding='utf-8') as json_file:
        #     hip_angle_data = json.load(json_file)
        # with open(knee_angle_json_path, mode='r', encoding='utf-8') as json_file:
        #     knee_angle_data = json.load(json_file)
        # with open(knee_to_hip_json_path, mode='r', encoding='utf-8') as json_file:
        #     knee_to_hip_data = json.load(json_file)
        with open(split_info_json_path, mode='r', encoding='utf-8') as json_file:
            split_info_data = json.load(json_file)
        
        # Write to DB only if a Recording ORM object is provided
        if recording is not None:
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
            # 'hip_angle': hip_angle_data,
            # 'knee_angle': knee_angle_data,
            # 'knee_to_hip': knee_to_hip_data,
            'split_info': split_info_data
        }
        return result
