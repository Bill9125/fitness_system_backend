from django.contrib import admin
from .models import Recording, Repetition, ErrorScore, RecommendedVideo

@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ('user', 'sport', 'total_frames', 'created_at')
    list_filter = ('sport', 'created_at')
    search_fields = ('user__username', 'sport')

@admin.register(Repetition)
class RepetitionAdmin(admin.ModelAdmin):
    list_display = ('id', 'recording', 'start_frame', 'end_frame')

@admin.register(ErrorScore)
class ErrorScoreAdmin(admin.ModelAdmin):
    list_display = ('rep', 'error', 'score')

@admin.register(RecommendedVideo)
class RecommendedVideoAdmin(admin.ModelAdmin):
    list_display = ('title', 'target_error', 'video_url')
