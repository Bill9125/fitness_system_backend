from django.urls import path
from . import views

urlpatterns = [
    path('bulk/', views.upload_bulk_videos, name='upload_bulk_videos'),
    path('single_to_session/', views.upload_single_video_to_session, name='upload_single_video_to_session'),
    path('sessions/', views.list_sessions, name='list_sessions'),
    path('single/', views.upload_single_video, name='upload_single_video'),
]
