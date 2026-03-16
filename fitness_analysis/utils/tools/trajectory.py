
import cv2
import os

def plot_trajectory(dir):
    # 載入影片和座標
    mp4_path = os.path.join(dir, 'vision_bar.mp4')
    avi_path = os.path.join(dir, 'vision_bar.avi')
    video_path = mp4_path if os.path.exists(mp4_path) else avi_path
    
    coordinates_path = os.path.join(dir, 'coordinates_interpolated.txt')
    output_path = os.path.join(dir, 'vision_bar_drawed.mp4')
    
    # 如果座標檔案不存在，則不處理
    if not os.path.exists(coordinates_path):

        print(f"[Trajectory] Coordinates file not found: {coordinates_path}")
        return

    # 解析座標檔案
    coordinates = {}
    with open(coordinates_path, 'r') as file:
        for line in file:
            if line.strip():  # 確保不處理空行
                data = line.strip().split(',')  # 用逗號分隔
                frame_number = int(data[0])    # 第一欄是 frame_number
                x = float(data[1])             # 第二欄是 x 座標
                y = float(data[2])             # 第三欄是 y 座標
                coordinates[frame_number] = (int(x), int(y))  # 存入座標字典

    # 打開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Trajectory] Could not open video: {video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用 mp4v 確保相容性
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 設定軌跡顯示的最近 N 幀（0 表示顯示全部軌跡）
    trajectory_window = 40 


    # 開始處理影片
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 取得軌跡點
        trajectory_points = []
        start_frame = 1 if trajectory_window == 0 else max(1, frame_count - trajectory_window)
        for i in range(start_frame, frame_count + 1):
            if i in coordinates:
                trajectory_points.append(coordinates[i])

        # 繪製軌跡（連線）
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                # 計算顏色深度（愈近目前的幀愈亮/深）
                # 如果 trajectory_window 為 0，則用總體進度計算漸層
                if trajectory_window > 0:
                    alpha = (i / len(trajectory_points))
                else:
                    alpha = (i / frame_count) if frame_count > 0 else 1.0
                
                # 顏色：從淺藍到深藍 (B, G, R)
                color = (255, int(150 * (1 - alpha)), int(100 * (1 - alpha)))
                
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], color, thickness=3, lineType=cv2.LINE_AA)

        # 寫入輸出影片
        out.write(frame)

    cap.release()
    out.release()
    print(f"[Trajectory] Trajectory video saved to {output_path}")

