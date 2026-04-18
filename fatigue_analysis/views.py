from django.http import JsonResponse, Http404
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
import os
from .models import Recording
from .utils.vertical_jump.yolo_inference import detect_jump_pose
from .utils.vertical_jump.analyzer import compute_vertical_jump

@extend_schema(
    summary="取得使用者的所有疲勞分析 Recording ID",
    responses={200: list}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_video_ids(request):
    """
    回傳當前使用者所有的疲勞分析紀錄清單。
    """
    user = request.user
    # 這裡的 Recording 指的是 fatigue_analysis.models.Recording
    recordings = Recording.objects.filter(user=user).order_by('-created_at')
    
    result = [
        {
            "id": r.id,
            "sport": r.sport,
            "tag": r.tag,
            "vjump_height": r.vjump_height,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in recordings
    ]
    
    return JsonResponse(result, safe=False)

@extend_schema(
    summary="取得垂直跳偵測結果",
    parameters=[
        OpenApiParameter(name='recording_id', type=OpenApiTypes.INT, location=OpenApiParameter.PATH, description='Recording ID')
    ],
    responses={200: dict}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_detection_result(request, recording_id: int):
    """
    計算或取得垂直跳高度。
    """
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        raise Http404("Recording not found")

    # 如果已經計算過，直接回傳
    if recording.vjump_height is not None:
        return JsonResponse({
            "recording_id": recording.id,
            "vjump_height": recording.vjump_height,
            "total_frames": recording.total_frames
        })

    # 執行分析邏輯
    video_path = os.path.join(settings.BASE_DIR, recording.folder, "video.mp4")
    if not os.path.exists(video_path):
        return JsonResponse({"error": f"Video file not found at {video_path}"}, status=404)

    # 1. 執行 YOLO 偵測取得關鍵點
    # 注意：這裡預設使用 yolo_inference.py 中的預設模型路徑
    try:
        kpts_data = detect_jump_pose(video_path)
        
        # 2. 執行垂直跳分析 (假設 FPS=60，您可以根據需求調整或從影片讀取)
        analysis_result = compute_vertical_jump(kpts_data, fps=60.0)
        
        # 3. 儲存結果
        recording.vjump_height = analysis_result['jump_height_cm']
        recording.total_frames = len(kpts_data)
        recording.save()
        
        return JsonResponse({
            "recording_id": recording.id,
            "vjump_height": recording.vjump_height,
            "total_frames": recording.total_frames,
            "details": analysis_result
        })
    except Exception as e:
        return JsonResponse({"error": f"Analysis failed: {str(e)}"}, status=500)

