from django.db import models
from django.contrib.auth.models import User

class VideoSession(models.Model):
    """
    Represents a collection of videos for a single exercise event.
    Can be a bulk upload (one user) or collaborative (multiple users).
    """
    sport = models.CharField(max_length=100, default='unknown_sport')
    folder = models.CharField(max_length=255, help_text="Relative path to the storage folder")
    session_token = models.CharField(max_length=10, null=True, blank=True, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_sessions')
    tag = models.CharField(max_length=100, default='unknown_tag')

    class Meta:
        db_table = 'video_sessions'

    def __str__(self):
        return f"Session {self.id} - {self.sport} (by {self.creator.username})"

class VideoFile(models.Model):
    """
    Individual video file record.
    """
    session = models.ForeignKey(VideoSession, on_delete=models.CASCADE, related_name='videos')
    uploader = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_videos')
    camera_angle = models.CharField(max_length=50) # e.g., 'front', 'side', 'top', or specific filenames
    original_filename = models.CharField(max_length=255)
    stored_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'video_files'

    def __str__(self):
        return f"Video {self.id} (Session {self.session_id}) - {self.camera_angle} by {self.uploader.username}"
