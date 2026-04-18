from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from datetime import datetime
import os
import shutil
from .models import VideoSession, VideoFile
from fitness_analysis.models import Recording as FitnessRecording
from fatigue_analysis.models import Recording as FatigueRecording
from .utils import reencode_video_on_upload, generate_random_id
from drf_spectacular.utils import extend_schema, OpenApiParameter
from rest_framework import serializers

# Serializers for API documentation
class BulkUploadSerializer(serializers.Serializer):
    sport = serializers.CharField(required=False, default='unknown_sport')
    bar_video = serializers.FileField(required=False)
    left_front_video = serializers.FileField(required=False)
    left_back_video = serializers.FileField(required=False)
    rear_video = serializers.FileField(required=False)
    top_video = serializers.FileField(required=False)
    tag = serializers.CharField(required=False, allow_blank=True)

class SingleUploadSerializer(serializers.Serializer):
    session_id = serializers.CharField(required=False)
    sport = serializers.CharField(required=False, default='unknown_sport')
    camera_angle = serializers.CharField(required=True)
    video = serializers.FileField(required=True)
    tag = serializers.CharField(required=False, allow_blank=True)

class SingleVideoUploadSerializer(serializers.Serializer):
    sport = serializers.CharField(required=True)
    tag = serializers.CharField(required=False, allow_blank=True)
    video = serializers.FileField(required=True)

@extend_schema(
    summary="多個影片同時上傳",
    description="由一個使用者一次上傳所有視角的影片。",
    request=BulkUploadSerializer,
    responses={200: dict}
)
@api_view(['POST', 'OPTIONS']) # 支援 OPTIONS 預檢請求
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_bulk_videos(request):
    if request.method == 'OPTIONS':
        return JsonResponse({"message": "OK"}, status=200)
    user = request.user
    sport = request.data.get('sport', 'unknown_sport')
    tag = request.data.get('tag', '')
    
    # Create directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"recordings/{sport.lower()}/recording_{timestamp}"
    folder_path = os.path.join(settings.BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Mapping for common angles
    file_mapping = {
        'bar_video': 'vision_bar.mp4',
        'left_front_video': 'vision_left-front.mp4',
        'left_back_video': 'vision_left-back.mp4',
        'rear_video': 'vision_rear.mp4',
        'top_video': 'vision_top.mp4',
    }
    
    recording = FitnessRecording.objects.create(
        user=user,
        sport=sport,
        folder=folder_name,
        tag=tag
    )
    
    uploaded_count = 0
    uploaded_files_info = []
    
    for file_key, target_name in file_mapping.items():
        uploaded_file = request.FILES.get(file_key)
        if uploaded_file:
            file_path = os.path.join(folder_path, target_name)
            with open(file_path, 'wb+') as dest:
                for chunk in uploaded_file.chunks():
                    dest.write(chunk)
            
            # Re-encode video
            reencode_video_on_upload(file_path)
            
            uploaded_count += 1
            uploaded_files_info.append({
                "angle": file_key,
                "stored_path": os.path.join(folder_name, target_name)
            })

    if uploaded_count == 0:
        recording.delete()
        shutil.rmtree(folder_path, ignore_errors=True)
        return JsonResponse({"error": "No valid video files uploaded."}, status=400)
        
    return JsonResponse({
        "message": "Bulk videos uploaded successfully",
        "recording_id": recording.id,
        "uploaded_count": uploaded_count,
        "files": uploaded_files_info,
        "folder": folder_name,
        "tag": tag
    })

@extend_schema(
    summary="單個影片上傳 for session",
    description="由不同使用者上傳不同視角的影片。實作方式是建立紀錄，將影片編號並記錄上傳者。",
    request=SingleUploadSerializer,
    responses={200: dict}
)
@api_view(['POST', 'OPTIONS']) # 支援 OPTIONS 預檢請求
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_single_video_to_session(request):
    if request.method == 'OPTIONS':
        return JsonResponse({"message": "OK"}, status=200)
    user = request.user
    session_id = request.data.get('session_id', None)
    camera_angle = request.data.get('camera_angle', 'unknown_angle')
    tag = request.data.get('tag', '')
    sport = request.data.get('sport', 'unknown_sport')
    uploaded_file = request.FILES.get('video')
    
    if not uploaded_file:
        return JsonResponse({"error": "No video file provided."}, status=400)
    
    if session_id:
        try:
            session = VideoSession.objects.get(session_token=session_id)
        except VideoSession.DoesNotExist:
            print(f"[DEBUG] Session token {session_id} not found, creating new session with this token")
            session = VideoSession.objects.create(
                sport=sport,
                folder=f"tmp_video/{session_id}",
                session_token=session_id,
                creator=user,
                tag=tag
            )
        
        # Check if this angle already exists in the session
        if VideoFile.objects.filter(session=session, camera_angle=camera_angle).exists():
            print(f"[DEBUG] Duplicate angle: {camera_angle} for session {session_id}")
            return JsonResponse({
                "error": f"Angle '{camera_angle}' has already been uploaded for this session."
            }, status=400)
        
        # Ensure session folder is correctly set (e.g. if we want to ensure tmp_video prefix)
        if not session.folder or not session.folder.startswith('tmp_video'):
            session.folder = f"tmp_video/{session_id}"
            session.save()
    else:
        print('create new session')
        # Auto-create session if not provided
        new_token = generate_random_id(5)
        session = VideoSession.objects.create(
            sport=sport,
            folder=f"tmp_video/{new_token}",
            session_token=new_token,
            creator=user,
            tag=tag
        )
        session_id = new_token # 更新以供後續回傳使用

    folder_path = os.path.join(settings.BASE_DIR, session.folder)
    os.makedirs(folder_path, exist_ok=True)
    
    # Store video in session folder
    # Prefix with uploader ID and angle for uniqueness
    target_name = f"vision_{camera_angle}_{user.id}_{datetime.now().strftime('%H%M%S')}.mp4"
    file_path = os.path.join(folder_path, target_name)
    
    with open(file_path, 'wb+') as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)
            
    reencode_video_on_upload(file_path)
    
    video_record = VideoFile.objects.create(
        session=session,
        uploader=user,
        camera_angle=camera_angle,
        original_filename=uploaded_file.name,
        stored_path=os.path.join(session.folder, target_name)
    )

    # 檢查該 Session 的視角是否上傳完成
    required_angles = []
    sport_lower = session.sport.lower()
    if 'deadlift' in sport_lower:
        required_angles = ['bar_video', 'left_front_video', 'left_back_video']
    elif 'benchpress' in sport_lower:
        required_angles = ['bar_video', 'rear_video', 'top_video']
    
    current_angles = list(session.videos.values_list('camera_angle', flat=True))
    missing_angles = [a for a in required_angles if a not in current_angles]
    is_completed = (len(required_angles) > 0 and len(missing_angles) == 0)

    recording_id = None
    if is_completed:
        # Move files to recordings folder
        final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_folder_name = f"recordings/{sport_lower}/recording_{final_timestamp}"
        final_folder_path = os.path.join(settings.BASE_DIR, final_folder_name)
        os.makedirs(final_folder_path, exist_ok=True)
        
        file_mapping = {
            'bar_video': 'vision_bar.mp4',
            'left_front_video': 'vision_left-front.mp4',
            'left_back_video': 'vision_left-back.mp4',
            'rear_video': 'vision_rear.mp4',
            'top_video': 'vision_top.mp4',
        }

        # 先取得所有影片紀錄以便搬移
        video_files = list(session.videos.all())

        for vf in video_files:
            src_path = os.path.join(settings.BASE_DIR, vf.stored_path)
            target_filename = file_mapping.get(vf.camera_angle, f"vision_{vf.camera_angle}.mp4")
            dst_path = os.path.join(final_folder_path, target_filename)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

        # Remove temporary directory
        old_folder_path = os.path.join(settings.BASE_DIR, session.folder)
        shutil.rmtree(old_folder_path, ignore_errors=True)

        # 備份必要資訊後刪除 Session (暫存表)
        session_creator = session.creator
        session_sport = session.sport
        session_tag = session.tag
        session.delete()

        # Create Recording for fitness_analysis compatibility
        recording = FitnessRecording.objects.create(
            user=session_creator,
            sport=session_sport,
            folder=final_folder_name,
            tag=session_tag
        )
        recording_id = recording.id

    return JsonResponse({
        "message": "Single video uploaded successfully",
        "session_id": session_id if not is_completed else None,
        "video_id": video_record.id if not is_completed else None,
        "uploader": user.username,
        "camera_angle": camera_angle,
        "stored_path": video_record.stored_path if not is_completed else None,
        "is_session_completed": is_completed,
        "tag": session_tag if is_completed else None,
        "missing_angles": missing_angles,
        "recording_id": recording_id
    })

@extend_schema(
    summary="單個影片直接上傳（不進入 Session）",
    description="直接上傳單個影片並建立 Recording 紀錄。",
    request=SingleVideoUploadSerializer,
    responses={200: dict}
)
@api_view(['POST', 'OPTIONS'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_single_video(request):
    if request.method == 'OPTIONS':
        return JsonResponse({"message": "OK"}, status=200)
    
    user = request.user
    sport = request.data.get('sport', 'unknown_sport')
    tag = request.data.get('tag', '')
    uploaded_file = request.FILES.get('video')
    
    if not uploaded_file:
        return JsonResponse({"error": "No video file provided."}, status=400)
    
    # Create directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"recordings/{sport.lower()}/single_{timestamp}"
    folder_path = os.path.join(settings.BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Store video
    target_name = f"video.mp4"
    file_path = os.path.join(folder_path, target_name)
    
    with open(file_path, 'wb+') as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)
            
    # Re-encode video
    reencode_video_on_upload(file_path)
    
    # Create Recording
    recording = FatigueRecording.objects.create(
        user=user,
        sport=sport,
        folder=folder_name,
        tag=tag
    )
    
    return JsonResponse({
        "message": "Single video uploaded successfully without session",
        "recording_id": recording.id,
        "sport": sport,
        "tag": tag,
        "folder": folder_name,
        "stored_path": os.path.join(folder_name, target_name)
    })

@extend_schema(
    summary="取得所有上傳階段與影片紀錄",
    responses={200: list}
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_sessions(request):
    sessions = VideoSession.objects.all().order_by('-created_at')
    result = []
    for s in sessions:
        videos = s.videos.all()
        result.append({
            "session_id": s.id,
            "sport": s.sport,
            "creator": s.creator.username,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "video_count": videos.count(),
            "videos": [
                {
                    "video_id": v.id,
                    "uploader": v.uploader.username,
                    "angle": v.camera_angle,
                    "original_name": v.original_filename,
                    "stored_path": v.stored_path,
                    "uploaded_at": v.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")
                } for v in videos
            ]
        })
    return JsonResponse(result, safe=False)
