from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import os
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO

from .common import rc_prep

class BaseProcessor(ABC):
    """
    Abstract Base Class for Fitness Analysis Processors.
    Handles video loading, main processing loop, and resource management.
    """
    def __init__(self):
        self.models = {}

    @abstractmethod
    def load_models(self):
        """
        Load necessary AI models (e.g., YOLO, Pose).
        """
        pass

    @abstractmethod
    def process_frame(self, frame_data: dict) -> Tuple[Any, Any]:
        """
        Process a single frame.
        
        Args:
            frame_data (dict): Contains 'frame', 'cap_index', 'models', etc.
            
        Returns:
            processed_frame: The frame with drawings (or None).
            frame_result: Data extracted from the frame (or None).
        """
        pass

    @abstractmethod
    def post_process(self, video_path: str):
        """
        Run specific post-processing logic (interpolation, charts, etc.)
        """
        pass

    def run(self, video_path: str):
        """
        Main execution pipeline.
        """
        first_time = time.time()
        
        # 1. Initialize Models
        self.load_models()
        
        # 2. Prepare Video I/O
        outs, bar_file, skeleton_files = rc_prep(video_path)
        print('[BaseProcessor] Video writers prepared.')
        
        # 3. Open Video Readers
        caps = self._open_captures(video_path)
        
        frame_count = 0
        frame_data_storage = {i: {} for i in range(len(caps))}
        
        start_time = time.time()
        try:
            while True:
                all_done = True
                
                for i, cap in enumerate(caps):
                    if cap is None: continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    all_done = False
                    
                    # Package data for processing
                    context = {
                        'frame': frame,
                        'cap_index': i,
                        'frame_count': frame_count,
                        'bar_file': bar_file if i == 0 else None,
                        'skeleton_files': skeleton_files
                    }
                    
                    # Delegate specific logic to subclass
                    processed_frame, frame_result = self.process_frame(context)
                    
                    # Store result
                    if frame_result:
                        # frame_result is typically a list of strings line by line
                        frame_data_storage[i][frame_count] = frame_result
                    
                    # Write video
                    if outs[i] and processed_frame is not None:
                        outs[i].write(processed_frame)

                frame_count += 1
                if all_done:
                    print("[BaseProcessor] All videos processed.")
                    print("Processing loop time:", time.time() - start_time)
                    break
                    
        finally:
            self._cleanup_resources(caps, outs, bar_file, skeleton_files, frame_data_storage, video_path)
            
        print('[BaseProcessor] Total run time:', time.time() - first_time)
        
        # 4. Post Processing
        self.post_process(video_path)
        
        return "Success"

    def _open_captures(self, video_path: str) -> List[cv2.VideoCapture]:
        caps = []
        # Support up to 3 views currently
        for i in range(3):
            mp4 = f'{video_path}/vision{i+1}.mp4'
            avi = f'{video_path}/vision{i+1}.avi'
            path = mp4 if os.path.exists(mp4) else avi
            
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[Warning] Failed to open camera {i+1}")
                caps.append(None)
            else:
                caps.append(cap)
        return caps

    def _cleanup_resources(self, caps, outs, bar_file, skeleton_files, data_storage, video_path=None):
        # Release Caps & Writers
        for cap in caps:
            if cap: cap.release()
        for out in outs:
            if out: out.release()

        # Write buffered data to files
        for i, frames_data in data_storage.items():
            buffer = []
            for _, lines in sorted(frames_data.items()):
                buffer.extend(lines)
            
            if i < len(skeleton_files) and skeleton_files[i]:
                skeleton_files[i].writelines(buffer)

        # Close files
        if bar_file: bar_file.close()
        for f in skeleton_files:
            if f: f.close()

        # 用 ffmpeg 重新編碼成 H.264 + faststart（瀏覽器才能串流播放）
        if video_path:
            import subprocess
            for i in range(len(outs)):
                raw_path = os.path.join(video_path, f'vision{i+1}_drawed.mp4')
                tmp_path = os.path.join(video_path, f'vision{i+1}_drawed.tmp.mp4')
                if os.path.exists(raw_path):
                    cmd = [
                        'ffmpeg', '-y', '-i', raw_path,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-movflags', '+faststart',
                        '-c:a', 'aac',
                        tmp_path
                    ]
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode == 0:
                        os.replace(tmp_path, raw_path)
                        print(f'[BaseProcessor] Re-encoded vision{i+1}_drawed.mp4 to H.264.')
                    else:
                        print(f'[BaseProcessor] ffmpeg failed for vision{i+1}_drawed.mp4: {result.stderr.decode()[-200:]}')
