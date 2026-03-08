from django.urls import path
from . import views
app_name = 'fitness'
urlpatterns = [
    path('upload/', views.upload_video, name='upload-video'),
    path('recordings/', views.get_user_recording_ids, name='get-user-recording-ids'),
    path('result/<int:recording_id>/', views.get_detection_result, name='get-detection-result'),
    path('videos/<int:recording_id>/<str:vision>/', views.get_videos, name='get-videos'),
    path('suggestion/<int:recording_id>/', views.get_suggestion, name='get-suggestion'),
    path('workout_plan/<int:recording_id>/', views.get_workout_plan, name='get-workout-plan'),
    path('recommendations/<int:recording_id>/', views.read_video_list, name='get-recommendations'),
]