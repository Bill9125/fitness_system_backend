from django.db import models
from django.contrib.auth.models import User

class Recording(models.Model):
    # 關聯到 Django 內建的 User
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recordings')
    sport = models.CharField(max_length=50, null=True, blank=True)
    folder = models.CharField(max_length=255, null=True, blank=True)
    total_frames = models.IntegerField(null=True, blank=True)
    training_suggestion = models.TextField(null=True, blank=True)
    workout_plan = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'recordings'

    def __str__(self):
        return f"{self.user.username} - {self.sport} ({self.created_at})"

class Repetition(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE, related_name='repetitions')
    start_frame = models.IntegerField(null=True, blank=True)
    end_frame = models.IntegerField(null=True, blank=True)
    error = models.CharField(max_length=255, null=True, blank=True)
    score = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'repetitions'

class RecommendedVideo(models.Model):
    title = models.CharField(max_length=255, null=True, blank=True)
    video_url = models.CharField(max_length=255, null=True, blank=True)
    target_error = models.CharField(max_length=255, null=True, blank=True)
    
    # 與 Recording 的多對多關聯，指定 through model 以控制 table 名稱與欄位
    recordings = models.ManyToManyField(
        Recording, 
        through='RecordingRecommendation',
        related_name='recommended_videos', 
        blank=True
    )

    class Meta:
        db_table = 'recommended_video'

class RecordingRecommendation(models.Model):
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE, db_column='recording_id')
    video = models.ForeignKey(RecommendedVideo, on_delete=models.CASCADE, db_column='video_id')

    class Meta:
        db_table = 'recording_recommendations'
        unique_together = ('recording', 'video')
