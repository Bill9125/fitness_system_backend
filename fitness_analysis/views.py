from django.http import JsonResponse, FileResponse, Http404
import os
import json
import cv2
import copy
import asyncio
import subprocess
import re
from .models import Recording, Repetition, ErrorScore
from .utils import DeadliftProcessor

def get_detection_result(request, pk: int):
    try:
        recording = Recording.objects.get(id=pk)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    repetitions = Repetition.objects.filter(recording=recording)
    error_scores = ErrorScore.objects.filter(rep__in=repetitions)
    
    folder = recording.folder
    if not error_scores.exists():
        processor = DeadliftProcessor()
        result = processor.run(folder)
        return JsonResponse({'result': result})

    # score_json_path = os.path.join(folder, "config/Score.json")
    # split_json_path = os.path.join(folder, "config/Split_info.json")

    # if not os.path.exists(score_json_path) or not os.path.exists(split_json_path):
    #     return JsonResponse({"error": "Config files not found"}, status=404)

    # with open(score_json_path, mode='r', encoding='utf-8') as json_file:
    #     score_data = json.load(json_file)['results']
    # with open(split_json_path, mode='r', encoding='utf-8') as json_file:
    #     split_data = json.load(json_file)
    
    # video_path = os.path.join(folder, "vision2.avi")
    # if not os.path.exists(video_path):
    #     video_path = os.path.join(folder, "vision2.mp4")
        
    # cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()

    # result = {'id': str(item_id), 'total_frames': total_frames, 'score': {}}
    # for idx, (key, item) in enumerate(score_data.items()):
    #     result['score'][key] = {}
    #     result['score'][key]['start_frame'] = split_data[key].get('start', 0)
    #     result['score'][key]['end_frame'] = split_data[key].get('end', 0)
    #     result['score'][key]['total_score'] = item[0]

    #     values = [item[1][0][1], item[1][1][1], item[1][2][1], item[1][3][1]]
    #     result['score'][key]['away_from_the_shins'] = values[0]
    #     result['score'][key]['hips_rise_before_barbell'] = values[1]
    #     result['score'][key]['colliding_with_the_knees'] = values[2]
    #     result['score'][key]['back_rounding'] = values[3]

    #     index_max = max(range(len(values)), key=values.__getitem__)
    #     if values[index_max] < 0.5:
    #         result['score'][key]['error'] = 'correct'
    #     else:
    #         errors = ["away_from_the_shins", "hips_rise_before_barbell", "colliding_with_the_knees", "back_rounding"]
    #         result['score'][key]['error'] = errors[index_max]

    # recording.total_frames = total_frames
    # recording.training_suggestion = json.dumps(result['score'])
    # recording.save()

    # return JsonResponse(result)

def get_graph(request, item_id: int):
    try:
        recording = Recording.objects.get(id=item_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")
        
    folder = recording.folder
    file_names = ['Bar_Position', 'Hip_Angle', 'Knee_Angle', 'Knee_to_Hip']
    result = []
    for file_name in file_names:
        json_path = os.path.join(folder, f"config/{file_name}.json")
        if os.path.exists(json_path):
            with open(json_path, mode='r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            result.append(data)
    return JsonResponse({'result': result})

async def get_videos(request, item_id: int, vision_index: int):
    try:
        # 在 async view 中使用同步 ORM 需要特別注意，或是改用 sync_to_async
        recording = await Recording.objects.aget(id=item_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")
        
    folder = recording.folder
    video_name = f"vision{vision_index}_drawed" if vision_index == 1 else f"vision{vision_index}_skeleton"
    avi_path = os.path.join(folder, f"{video_name}.avi")
    mp4_path = os.path.join(folder, f"{video_name}.mp4")

    # 等待 AVI 出現
    count = 0
    while not os.path.exists(avi_path):
        if count == 100:
            break
        count += 1
        await asyncio.sleep(1.5)

    if os.path.exists(mp4_path):
        return FileResponse(open(mp4_path, 'rb'), content_type="video/mp4")

    # 執行轉檔
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, avi_to_mp4, folder, video_name)

    return FileResponse(open(mp4_path, 'rb'), content_type="video/mp4")

def avi_to_mp4(folder, video_name):
    avi_path = os.path.join(folder, f"{video_name}.avi")
    mp4_path = os.path.join(folder, f"{video_name}.mp4")
    temp_path = os.path.join(folder, f"{video_name}.temp.mp4")

    if not os.path.exists(mp4_path):
        ffmpeg_cmd = [
            "ffmpeg", "-i", avi_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-strict", "experimental", temp_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        os.rename(temp_path, mp4_path)
    return mp4_path

def read_feedback(request, item_id: int):
    try:
        recording = Recording.objects.get(id=item_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")
        
    error = recording.training_suggestion # 假設存這裡
    
    # 這裡需要 api_func 的定義，若沒有會報錯
    # feedback_prompt = f"..."
    # feedback_result = api_func.get_openai_response(feedback_prompt)
    # return JsonResponse({'result': extract_markdown(feedback_result)})
    
    return JsonResponse({'error': 'api_func not implemented'}, status=501)

def extract_markdown(llm_output: str) -> str:
    return re.sub(r'^```(?:markdown)?\n([\s\S]+?)\n```$', r'\1', llm_output.strip())