import os
import cv2

def rc_prep(video_path: str):
    """
    Prepare output video writers and data files.
    
    Args:
        video_path: Path to the directory containing input videos.
        
    Returns:
        outs: List of cv2.VideoWriter objects for the 3 views.
        bar_file: File object for writing bar detection data.
        skeleton_files: List of file objects for writing skeleton data for each view.
    """
    os.makedirs(f'{video_path}/config', exist_ok=True)
    
    # 1. Prepare Video Writers
    outs = []
    views = ['bar', 'left-front', 'left-back']  # default
    if 'deadlift' in video_path:
        views = ['bar', 'left-front', 'left-back']
    elif 'benchpress' in video_path:
        views = ['bar', 'rear', 'top']
        
    # We assume 3 views as per the original code logic
    for view in views:
        input_path_mp4 = f'{video_path}/vision_{view}.mp4'
        input_path_avi = f'{video_path}/vision_{view}.avi'
        
        cap = None
        if os.path.exists(input_path_mp4):
            cap = cv2.VideoCapture(input_path_mp4)
        elif os.path.exists(input_path_avi):
            cap = cv2.VideoCapture(input_path_avi)
            
        if cap and cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Using mp4v codec for mp4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = f'{video_path}/vision_{view}_drawed.mp4'
            
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            outs.append(out)
            cap.release()
        else:
            print(f"Warning: Could not open input video for vision_{view} to initialize writer.")
            outs.append(None)

    # 2. Prepare Data Files
    # Bar data file - assuming it's only relevant for the first view or shared
    bar_file_path = f'{video_path}/coordinates.txt'
    bar_file = open(bar_file_path, 'w')
    
    # Skeleton data files for each view
    skeleton_files = []
    for v in views:
        if v == 'bar' and 'benchpress' in video_path:
            skeleton_files.append(None)
            continue
        sk_path = f'{video_path}/skeleton_{v}.txt'
        f = open(sk_path, 'w')
        skeleton_files.append(f)
        
    return outs, bar_file, skeleton_files
