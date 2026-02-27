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
from django.conf import settings

class BenchpressProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.bar_model_path = settings.BAR_MODEL_PATH
        self.rear_pose_model_path = settings.BENCHPRESS_REAR_POSE_MODEL_PATH
        self.top_pose_model_path = settings.BENCHPRESS_TOP_POSE_MODEL_PATH
        self.skeleton_connections = SKELETON_CONNECTIONS

    def load_models(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.models['bar'] = YOLO(self.bar_model_path).to(device)
        self.models['rear_pose'] = YOLO(self.rear_pose_model_path).to(device)
        self.models['top_pose'] = YOLO(self.top_pose_model_path).to(device)
        
        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            self.models['bar'].predict([dummy], verbose=False)
            self.models['rear_pose'].predict([dummy], verbose=False)
            self.models['top_pose'].predict([dummy], verbose=False)
            
        print(f"[BenchpressProcessor] Models loaded on {device}")

    def process_frame(self, frame_data: dict) -> Tuple[Any, Any]:
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
                self.skeleton_connections,
                bar_file,
                frame_cnt,
                bone_model=None,
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
                self.skeleton_connections, 
                frame_cnt
            )
            return processed_frame, sk_list
        
        elif idx == 2:
            # Side Views: Pose Only
            processed_frame, sk_list = bone_frame(
                frame, 
                self.models['top_pose'], 
                self.skeleton_connections, 
                frame_cnt
            )
            return processed_frame, sk_list
        
        else:
            raise ValueError(f"Invalid idx: {idx}")

    def post_process(self, video_path: str):
        # Run the standard pipeline
        pass

    def run(self, video_path: str):
        pass
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
# f'python ./tools/Benchpress_tool/step6_cut.py "{folder}"',
# f'python ./tools/Benchpress_tool/step7_length_100.py "{folder}"',
# f'python ./tools/Benchpress_tool/step8_normalize.py "{folder}"',