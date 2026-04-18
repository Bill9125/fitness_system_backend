import os
import subprocess

def reencode_video_on_upload(file_path):
    """
    使用 ffmpeg 將上傳的影片統一轉碼為 H.264/AAC 格式，
    確保分析管線與前端播放的相容性。
    """
    tmp_path = file_path + ".tmp.mp4"
    cmd = [
        'ffmpeg', '-y', '-i', file_path,
        '-r', '30',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-movflags', '+faststart',
        '-c:a', 'aac',
        tmp_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp_path, file_path)
        print(f"[Upload] Re-encoded {file_path} successfully.")
    except Exception as e:
        print(f"[Upload] Re-encoding failed for {file_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
def generate_random_id(length=5):
    """生成 5 位英數字混雜的隨機 ID"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))