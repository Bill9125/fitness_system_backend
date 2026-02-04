from django.conf import settings

class DeadliftProcessor:
    def __init__(self):
        self.bar_model_path = settings.BAR_MODEL_PATH
        self.error_model_path = settings.DEADLIFT_ERROR_MODEL_PATH
        self.pose_model_path = settings.DEADLIFT_POSE_MODEL_PATH
    def run(self, folder: str):
        return folder