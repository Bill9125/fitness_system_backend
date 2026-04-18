from django.http import JsonResponse, FileResponse, Http404, StreamingHttpResponse
import os
import json
import time
import re
import ast
from .models import Recording, Repetition, RecommendedVideo, RecordingRecommendation
from .utils import ProcessorFactory, OpenAIClient
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny

@extend_schema(
    summary="取得使用者的所有 Recording ID",
    responses={200: dict}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_recording_ids(request):
    # 直接回傳 list 格式，讓前端方便處理
    user_id = request.user.id
    recordings = Recording.objects.filter(user=user_id)
    result = [
        {
            "id": r.id,
            "sport": r.sport,
            "tag": r.tag,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } 
        for r in recordings
    ]
    return JsonResponse(result, safe=False)

@extend_schema(
    summary="取得動作偵測結果（含分數、角度、分段資訊），如果已經有predict過的會直接回傳，不會再重新predict",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: dict}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_detection_result(request, recording_id: int):
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")
    repetitions = Repetition.objects.filter(recording=recording)
    folder = recording.folder
    sport_type = recording.sport
    processor = ProcessorFactory.get_processor(sport_type)
    if not repetitions.exists():
        start_time = time.time()
        processor.run(folder)
        end_time = time.time()
        print(f"[get_detection_result] Processing time: {end_time - start_time}")
    result = processor.get_result(folder, recording=recording)
    if not result:
        return JsonResponse({"error": "Config files not found"}, status=404)
    return JsonResponse(result) 

@extend_schema(
    summary="取得影片",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID'),
        OpenApiParameter(name='vision', type=OpenApiTypes.STR, location=OpenApiParameter.PATH, description='Vision')
    ],
    responses={200: dict}
)
@api_view(['GET', 'HEAD', 'OPTIONS'])
@permission_classes([AllowAny])
def get_videos(request, recording_id: int, vision: str):
    # 這裡的 get_videos 不能用 async def 若用了 api_view 裝飾器。先移除 async 標籤或改回原版方式限制，但 DRF 逐漸支援 async。
    # 簡化為 def:
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    from django.conf import settings
    # 將相對路徑轉為絕對路徑，避免 CWD 不同造成 os.path.exists 失敗
    folder = os.path.join(settings.BASE_DIR, recording.folder)
    video_name = f"vision_{vision}_drawed"
    mp4_path = os.path.join(folder, f"{video_name}.mp4")

    print(f"[get_videos] Looking for: {mp4_path}, exists: {os.path.exists(mp4_path)}")

    if not os.path.exists(mp4_path):
        return JsonResponse({"error": f"Video not found: {mp4_path}"}, status=404)

    file_size = os.path.getsize(mp4_path)
    range_header = request.headers.get('Range', None)

    import mimetypes
    content_type, _ = mimetypes.guess_type(mp4_path)
    if not content_type:
        content_type = 'video/mp4'

    if range_header:
        match = re.search(r'bytes=(\d+)-(\d+)?', range_header)
        if match:
            start = int(match.group(1))
            end_match = match.group(2)
            end = int(end_match) if end_match else file_size - 1

            if start >= file_size:
                return JsonResponse({"error": "Range Not Satisfiable"}, status=416)

            chunk_size = (end - start) + 1

            # In Django, FileResponse naturally handles file seeking
            file = open(mp4_path, 'rb')
            file.seek(start)
            
            # The custom iterator helps us read only `chunk_size` bytes
            def file_iterator(file_obj, chunk, blk_size=8192):
                remaining = chunk
                while remaining > 0:
                    read_size = min(blk_size, remaining)
                    data = file_obj.read(read_size)
                    if not data:
                        break
                    yield data
                    remaining -= len(data)
                file_obj.close()
            
            response = StreamingHttpResponse(file_iterator(file, chunk_size), status=206, content_type=content_type)
            response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response['Content-Length'] = str(chunk_size)
            response['Accept-Ranges'] = 'bytes'
            response['Cache-Control'] = 'no-cache'
            return response

    response = FileResponse(open(mp4_path, 'rb'), content_type=content_type)
    response['Content-Length'] = str(file_size)
    response['Accept-Ranges'] = 'bytes'
    response['Cache-Control'] = 'no-cache'
    return response

@extend_schema(
    summary="取得 AI 健身建議（串流形式）",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: str}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
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
@permission_classes([IsAuthenticated])
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
@permission_classes([IsAuthenticated])
def get_recommendations(request, recording_id: int):
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    # 優先從資料庫撈取已儲存的推薦影片
    recommend_videos = recording.recommended_videos.all()
    if recommend_videos.exists():
        result = [
            {
                "title": v.title,
                "url": v.video_url,
                "video_id": v.video_url.split("v=")[-1],
                "target_error": v.target_error
            }
            for v in recommend_videos
        ]
        return JsonResponse({'result': result})

    # 取得當前拍攝紀錄的所有錯誤
    repetitions = recording.repetitions.all()
    all_errors = set()
    for rep in repetitions:
        if rep.error:
            # rep.error 可能包含多個以逗號隔開的錯誤：'error1,error2,'
            errs = [e.strip() for e in rep.error.split(',') if e.strip()]
            all_errors.update(errs)
    
    # 將所有錯誤整合為一個字串供提示詞使用
    error_context = ', '.join(all_errors)
    if not error_context:
        # 如果沒有偵測到錯誤，可以回傳空 list 或預設提示
        return JsonResponse({'result': []})

    movement_list = ['平板支撐', '臀推', '深蹲', '傳統硬舉', '羅馬尼亞硬舉', '臥推', '引體向上', '肩推']
    recommend_video_prompt = """你是一個健身教練，這是我在做一組訓練時發生的錯誤 -- %s。你會建議我針對哪幾種動作進行優化或自主練習？請從這份清單中挑選: %s。請回覆我一個格式如下的 list of strings: ["動作A", "動作B"]。請只回覆該串列資料即可，不要有其他描述言論。""" % (error_context, ', '.join(movement_list))

    openai_client = OpenAIClient()
    times = 0
    result = []
    
    while True:
        try:
            ai_response = openai_client.get_response(recommend_video_prompt)
            # 使用正則抓取中括號內容
            match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not match:
                raise ValueError("AI response does not contain a list")
            
            # 使用 ast.literal_eval 將字串轉為真正的 list
            selected_movements = ast.literal_eval(match.group(0))
            
            for movement_name in selected_movements:
                # 從資料庫中尋找對應的 RecommendedVideo
                videos = RecommendedVideo.objects.filter(target_error=movement_name)
                for v in videos:
                    # 建立關聯 (透過 through model)
                    recording.recommended_videos.add(v)
                    result.append({
                        "title": v.title,
                        "url": v.video_url,
                        "video_id": v.video_url.split("v=")[-1],
                        "target_error": v.target_error
                    })
            break
        except Exception as e:
            print(f"[get_recommendations] AI parsing error (trial {times+1}): {e}")
            times += 1
            if times >= 5:
                break
            continue
    return JsonResponse({'result': result})
