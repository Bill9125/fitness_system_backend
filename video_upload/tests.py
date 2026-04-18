import os
import shutil
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APITestCase
from rest_framework import status
from .models import VideoSession, VideoFile
from unittest.mock import patch

# 定義測試儲存路徑
TEST_BASE_DIR = '/home/rinku_wu/workspace/fitness_system_backend/test_storage'

@override_settings(BASE_DIR=TEST_BASE_DIR)
class VideoUploadTestCase(APITestCase):
    def setUp(self):
        # 建立測試使用者
        self.user = User.objects.create_user(username='tester', password='password123')
        self.other_user = User.objects.create_user(username='tester2', password='password123')
        
        # 登入為測試使用者
        self.client.force_authenticate(user=self.user)
        
        # 確保測試目錄存在
        if not os.path.exists(TEST_BASE_DIR):
            os.makedirs(TEST_BASE_DIR)

    def tearDown(self):
        # 清理測試目錄
        if os.path.exists(TEST_BASE_DIR):
            shutil.rmtree(TEST_BASE_DIR, ignore_errors=True)

    @patch('video_upload.utils.reencode_video_on_upload')
    def test_upload_bulk_videos_success(self, mock_reencode):
        """測試一次上傳多個影片 (Bulk Upload)"""
        url = reverse('upload_bulk_videos')
        
        video_content = b"dummy_content"
        data = {
            'sport': 'benchpress',
            'bar_video': SimpleUploadedFile('bar.mp4', video_content, content_type='video/mp4'),
            'rear_video': SimpleUploadedFile('rear.mp4', video_content, content_type='video/mp4'),
        }
        
        response = self.client.post(url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # response should contain session_id and uploaded_count
        res_data = response.json()
        self.assertEqual(res_data['uploaded_count'], 2)
        self.assertEqual(VideoSession.objects.count(), 1)
        self.assertEqual(VideoFile.objects.filter(session_id=res_data['session_id']).count(), 2)

    @patch('video_upload.utils.reencode_video_on_upload')
    def test_upload_bulk_videos_no_files(self, mock_reencode):
        """測試在沒有檔案的情況下上傳"""
        url = reverse('upload_bulk_videos')
        data = {'sport': 'benchpress'}
        
        response = self.client.post(url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('No valid video files uploaded', response.json()['error'])
        self.assertEqual(VideoSession.objects.count(), 0)

    @patch('video_upload.utils.reencode_video_on_upload')
    def test_upload_single_video_new_session(self, mock_reencode):
        """測試單個影片上傳 (新的工作階段)"""
        url = reverse('upload_single_video')
        video_content = b"dummy_content"
        data = {
            'sport': 'deadlift',
            'camera_angle': 'left_front_video',
            'video': SimpleUploadedFile('video.mp4', video_content, content_type='video/mp4'),
        }
        
        response = self.client.post(url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        res_data = response.json()
        self.assertIn('session_id', res_data)
        
        # 驗證資料夾路徑是否正確
        session = VideoSession.objects.get(id=res_data['session_id'])
        self.assertEqual(session.folder, f"tmp_video/{session.id}")
        self.assertTrue(os.path.exists(os.path.join(TEST_BASE_DIR, session.folder)))

    @patch('video_upload.utils.reencode_video_on_upload')
    def test_upload_single_video_existing_session(self, mock_reencode):
        """測試單個影片上傳 (上傳至現有的工作階段)"""
        # 先建立一個 session
        session = VideoSession.objects.create(sport='deadlift', creator=self.user, folder='tmp_video/123')
        
        url = reverse('upload_single_video')
        video_content = b"dummy_content"
        
        # 此時是另一個使用者上傳
        self.client.force_authenticate(user=self.other_user)
        data = {
            'session_id': session.id,
            'camera_angle': 'bar_video',
            'video': SimpleUploadedFile('bar.mp4', video_content, content_type='video/mp4'),
        }
        
        response = self.client.post(url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(VideoFile.objects.filter(session=session, camera_angle='bar_video').count(), 1)
        # 驗證錄影者是否為 other_user
        video_file = VideoFile.objects.get(session=session, camera_angle='bar_video')
        self.assertEqual(video_file.uploader, self.other_user)

    @patch('video_upload.utils.reencode_video_on_upload')
    def test_upload_single_video_duplicate_angle_error(self, mock_reencode):
        """測試重複上傳相同視角時應報錯"""
        # 建立 session 並上傳一個視角
        session = VideoSession.objects.create(sport='deadlift', creator=self.user, folder='tmp_video/123')
        VideoFile.objects.create(
            session=session, 
            uploader=self.user, 
            camera_angle='bar_video', 
            stored_path='...', 
            original_filename='...'
        )
        
        url = reverse('upload_single_video')
        video_content = b"dummy_content"
        data = {
            'session_id': session.id,
            'camera_angle': 'bar_video',
            'video': SimpleUploadedFile('bar.mp4', video_content, content_type='video/mp4'),
        }
        
        response = self.client.post(url, data, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("already been uploaded", response.json()['error'])

    def test_list_sessions(self):
        """測試列出所有工作階段與影片"""
        # 準備資料
        session = VideoSession.objects.create(sport='squat', creator=self.user, folder='tmp_storage/321')
        VideoFile.objects.create(
            session=session, 
            uploader=self.user, 
            camera_angle='angle1', 
            stored_path='...', 
            original_filename='file1.mp4'
        )
        
        url = reverse('list_sessions')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        res_data = response.json()
        self.assertTrue(len(res_data) >= 1)
        self.assertEqual(res_data[0]['sport'], 'squat')
        self.assertEqual(len(res_data[0]['videos']), 1)
