import threading
import subprocess

def run_scripts():
    scripts = [
        "step0_hampel_bar.py",
        "step0_hampel_yolo_ske_rear.py",
        "step0_hampel_yolo_ske_top.py",
        "step1_interpolate_bar.py",
        "step2_interpolate_yolo_ske.py",
        "step3_autocutting_0801.py",
        "step5_calculate_angle_new_feature_test.py",
        "step6_cut.py",
        "step7_length_100.py",
        "step8_normalize.py"
    ]
    for script in scripts:
        print(f"🚀 執行 {script} 中...")
        subprocess.run(["python", script])

# 建立一個 thread 來跑這些腳本
t = threading.Thread(target=run_scripts)
t.start()

# 主程式可以繼續執行其他任務
print("🧵 主程式繼續執行其他事情...")
