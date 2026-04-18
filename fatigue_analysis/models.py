from django.db import models
from django.contrib.auth.models import User

class Recording(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='fatigue_recordings')
    sport = models.CharField(max_length=50, null=True, blank=True)
    folder = models.CharField(max_length=255, null=True, blank=True)
    total_frames = models.IntegerField(null=True, blank=True)
    vjump_height = models.FloatField(null=True, blank=True)
    tag = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'recordings_fatigue'

    def __str__(self):
        return f"{self.user.username} - {self.sport} Fatigue ({self.created_at})"
