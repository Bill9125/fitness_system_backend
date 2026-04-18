from django.urls import path
from . import views

urlpatterns = [
    path('recordings/', views.get_user_video_ids, name='get_user_video_ids'),
    path('result/<int:recording_id>/', views.get_detection_result, name='get_detection_result'),
]
