from django.urls import path
from . import views
app_name = 'fitness'
urlpatterns = [
    # 現在網址會變成 api/analysis/1/
    path('<int:pk>/result', views.get_detection_result, name='get-detection-result'), 
]