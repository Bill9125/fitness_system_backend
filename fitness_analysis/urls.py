from django.urls import path
from . import views
urlpatterns = [
    # catslab.ee.ncku.edu.tw:9125/api/analysis/list
    path('list/', views.get_recordings, name='recording-list'),
]