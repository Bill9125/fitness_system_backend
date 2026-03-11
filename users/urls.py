from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from .views import RegisterView, UserProfileView, SendVerificationEmailView
from .serializers import EmailTokenObtainSerializer
from drf_spectacular.utils import extend_schema, extend_schema_view

# Custom view forcing strictly Email checks
class EmailTokenObtainPairView(TokenObtainPairView):
    serializer_class = EmailTokenObtainSerializer

# 裝飾登入與刷新 API 用以產生正確的 Swagger 文件
LoginView = extend_schema_view(
    post=extend_schema(summary="登入並獲取 Access Token", description="驗證信箱與密碼並發放一組長短效 JWT Access Token 及 Refresh Token。")
)(EmailTokenObtainPairView)

RefreshView = extend_schema_view(
    post=extend_schema(summary="刷新 Access Token", description="當 Access Token 過期時，使用此 API 夾帶有效的 Refresh Token 換取一組新的 Access Token。")
)(TokenRefreshView)

urlpatterns = [
    path('login/', LoginView.as_view(), name='token_obtain_pair'),
    path('refresh/', RefreshView.as_view(), name='token_refresh'),
    path('send-verification/', SendVerificationEmailView.as_view(), name='send-verification'),
    path('register/', RegisterView.as_view(), name='auth_register'),
    path('profile/', UserProfileView.as_view(), name='user_profile'),
]
