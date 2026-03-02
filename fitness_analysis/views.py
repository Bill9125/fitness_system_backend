from django.http import JsonResponse, FileResponse, Http404, StreamingHttpResponse
import os
import json
import time
from .models import Recording, Repetition, RecommendedVideo
from .utils import ProcessorFactory, OpenAIClient
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from rest_framework.decorators import api_view
from rest_framework.response import Response

@extend_schema(
    summary="取得使用者的所有 Recording ID",
    responses={200: dict}
)
@api_view(['GET'])
def get_user_recording_ids(request, user_id: int):
    try:
        recordings = Recording.objects.filter(user=user_id)
        recording_ids = [recording.id for recording in recordings]
        return JsonResponse({'recording_ids': recording_ids})
    except Recording.DoesNotExist:
        raise Http404("User not found")
    

@extend_schema(
    summary="取得動作偵測結果（含分數、角度、分段資訊），如果已經有predict過的會直接回傳，不會再重新predict",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: dict}
)
@api_view(['GET'])
def get_detection_result(request, recording_id: int):
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")
    repetitions = Repetition.objects.filter(recording=recording)
    folder = recording.folder
    result = {}
    if repetitions.exists():
        sport_type = recording.sport
        processor = ProcessorFactory.get_processor(sport_type)
        start_time = time.time()
        processor.run(folder)
        end_time = time.time()
        print(f"[get_detection_result] Processing time: {end_time - start_time}")

    score_json_path = os.path.join(folder, "config/Score.json")
    bar_position_json_path = os.path.join(folder, "config/Bar_Position.json")
    hip_angle_json_path = os.path.join(folder, "config/Hip_Angle.json")
    knee_angle_json_path = os.path.join(folder, "config/Knee_Angle.json")
    knee_to_hip_json_path = os.path.join(folder, "config/Knee_to_Hip.json")
    split_info_json_path = os.path.join(folder, "config/Split_info.json")

    if not os.path.exists(score_json_path):
        return JsonResponse({"error": "Config files not found"}, status=404)
    with open(score_json_path, mode='r', encoding='utf-8') as json_file:
        score_data = json.load(json_file)['results']
    with open(bar_position_json_path, mode='r', encoding='utf-8') as json_file:
        bar_position_data = json.load(json_file)
    with open(hip_angle_json_path, mode='r', encoding='utf-8') as json_file:
        hip_angle_data = json.load(json_file)
    with open(knee_angle_json_path, mode='r', encoding='utf-8') as json_file:
        knee_angle_data = json.load(json_file)
    with open(knee_to_hip_json_path, mode='r', encoding='utf-8') as json_file:
        knee_to_hip_data = json.load(json_file)
    with open(split_info_json_path, mode='r', encoding='utf-8') as json_file:
        split_info_data = json.load(json_file)
        for key, val in split_info_data.items():
            rep_score = score_data.get(key, [0])["score"]
            errors = ''
            for error, conf in score_data[key].items():
                if error != "score" and conf >= 0.5:
                    errors += error + ','
            Repetition.objects.update_or_create(
                recording=recording,
                start_frame=val.get('start'),
                end_frame=val.get('end'),
                score=rep_score,
                error=errors
            )

    result = {
        'score': score_data,
        'bar_position': bar_position_data,
        'hip_angle': hip_angle_data,
        'knee_angle': knee_angle_data,
        'knee_to_hip': knee_to_hip_data,
        'split_info': split_info_data
    }
    return JsonResponse(result)

# 合併進 get_detection_result 裡面
# def get_graph(request, item_id: int):
#     try:
#         recording = Recording.objects.get(id=item_id)
#     except Recording.DoesNotExist:
#         raise Http404("Recording not found")
    
#     folder = recording.folder
#     file_names = ['Bar_Position', 'Hip_Angle', 'Knee_Angle', 'Knee_to_Hip']
#     result = []
#     for file_name in file_names:
#         json_path = os.path.join(folder, f"config/{file_name}.json")
#         if os.path.exists(json_path):
#             with open(json_path, mode='r', encoding='utf-8') as json_file:
#                 data = json.load(json_file)
#             result.append(data)
#     return JsonResponse({'result': result})

@extend_schema(
    summary="取得影片",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID'),
        OpenApiParameter(name='vision_index', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Vision Index')
    ],
    responses={200: dict}
)
async def get_videos(request, recording_id: int, vision_index: int):
    # 注意：async view 不能用 @api_view（DRF 不支援），直接用 Django 原生 async view
    try:
        recording = await Recording.objects.aget(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    from django.conf import settings
    # 將相對路徑轉為絕對路徑，避免 CWD 不同造成 os.path.exists 失敗
    folder = os.path.join(settings.BASE_DIR, recording.folder)
    video_name = f"vision{vision_index}_drawed"
    mp4_path = os.path.join(folder, f"{video_name}.mp4")

    print(f"[get_videos] Looking for: {mp4_path}, exists: {os.path.exists(mp4_path)}")

    if os.path.exists(mp4_path):
        return FileResponse(open(mp4_path, 'rb'), content_type="video/mp4")

    return JsonResponse({"error": f"Video not found: {mp4_path}"}, status=404)

# def avi_to_mp4(folder, video_name):
#     avi_path = os.path.join(folder, f"{video_name}.avi")
#     mp4_path = os.path.join(folder, f"{video_name}.mp4")
#     temp_path = os.path.join(folder, f"{video_name}.temp.mp4")

#     if not os.path.exists(mp4_path):
#         ffmpeg_cmd = [
#             "ffmpeg", "-i", avi_path,
#             "-c:v", "libx264", "-c:a", "aac",
#             "-strict", "experimental", temp_path
#         ]
#         subprocess.run(ffmpeg_cmd, check=True)
#         os.rename(temp_path, mp4_path)
#     return mp4_path

@extend_schema(
    summary="取得 AI 健身建議（串流形式）",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: str}
)
@api_view(['GET'])
def get_suggestion(request, recording_id: int):
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    if recording.training_suggestion:
        def stored_suggestion_gen():
            yield f"data: {json.dumps({'data': recording.training_suggestion})}\n\n"
            yield f"data: {json.dumps({'event': 'end', 'data': ''})}\n\n"
        return StreamingHttpResponse(stored_suggestion_gen(), content_type="text/event-stream")

    openai_client = OpenAIClient()
    gen = openai_client.get_suggestion_stream(recording_id)

    def save_suggestion_wrapper(generator):
        full_content = ""
        for chunk_str in generator:
            yield chunk_str
            try:
                clean_chunk = chunk_str.strip()
                if clean_chunk.startswith("data: "):
                    data_obj = json.loads(clean_chunk[6:])
                    if "data" in data_obj and "event" not in data_obj:
                        full_content += data_obj["data"]
            except:
                pass
        
        if full_content:
            recording.training_suggestion = OpenAIClient.extract_markdown(full_content)
            recording.save()

    return StreamingHttpResponse(save_suggestion_wrapper(gen), content_type="text/event-stream")

@extend_schema(
    summary="取得 AI 訓練菜單（串流形式）",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: str}
)
@api_view(['GET'])
def get_workout_plan(request, recording_id: int):
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    if recording.workout_plan:
        def stored_plan_gen():
            yield f"data: {json.dumps({'data': recording.workout_plan})}\n\n"
            yield f"data: {json.dumps({'event': 'end', 'data': ''})}\n\n"
        return StreamingHttpResponse(stored_plan_gen(), content_type="text/event-stream")

    openai_client = OpenAIClient()
    gen = openai_client.get_workout_plan_stream(recording_id)

    def save_plan_wrapper(generator):
        full_content = ""
        for chunk_str in generator:
            yield chunk_str
            try:
                clean_chunk = chunk_str.strip()
                if clean_chunk.startswith("data: "):
                    data_obj = json.loads(clean_chunk[6:])
                    if "data" in data_obj and "event" not in data_obj:
                        full_content += data_obj["data"]
            except:
                pass
        
        if full_content:
            recording.workout_plan = OpenAIClient.extract_markdown(full_content)
            recording.save()

    return StreamingHttpResponse(save_plan_wrapper(gen), content_type="text/event-stream")

@extend_schema(
    summary="取得 AI 推薦影片",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: str}
)
@api_view(['GET'])
def read_video_list(recording_id: int):
    recording = Recording.objects.get(id=recording_id)
    if not recording or "error" not in recording:
        raise HTTPException(status_code=404, detail="Recording or error feedback not found")
    error = recording["error"]
    with open("video_list.json", mode='r', encoding='utf-8') as json_file:
        movement = json.load(json_file)

    movement_list = ['平板支撐', '臀推', '深蹲', '傳統硬舉', '羅馬尼亞硬舉', '臥推', '引體向上', '肩推']
    recommend_video_prompt = """你是一個健身教練，這是我在做一組硬舉訓練時發生的錯誤--%s，你會建議我這些動作裡的哪些%s？請回我一個格式一樣的 list of string""" % (error, ' '.join(movement_list))
    times = 0
    while True:
        result = []
        try:
            videos = api_func.get_openai_response(recommend_video_prompt)
            print(videos)
            # 用正則抓中括號內部的資料
            match = re.search(r'\[.*\]', videos, re.DOTALL)
            list_text = match.group(0)
            # 用 ast.literal_eval 將字串轉為真正的 list of dict
            video_list = ast.literal_eval(list_text)
            for text in video_list:
                if text in movement_list:
                    video = movement[text][0]
                    video["video_id"] = video['url'].split("v=")[-1]
                    result.append(video)
            break
        except:
            times += 1
            if times >= 5:
                break
            continue

    return {'result': result}
