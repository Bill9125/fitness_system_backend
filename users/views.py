from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_spectacular.utils import extend_schema, extend_schema_view
from django.core.mail import send_mail
from django.conf import settings
import random

from .models import EmailVerificationToken, UserProfile
from .serializers import (
    SendVerificationSerializer,
    RegisterSerializer,
    UserSerializer,
    FillProfileSerializer
)

class SendVerificationEmailView(generics.CreateAPIView):
    permission_classes = (AllowAny,)
    serializer_class = SendVerificationSerializer

    @extend_schema(summary="發送驗證碼信件", description="前端傳入 email，後端將產生 6 位數驗證碼寄出。")
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']

        # Generate OTP
        code = str(random.randint(100000, 999999))
        
        # Save or update Token
        EmailVerificationToken.objects.update_or_create(
            email=email,
            defaults={'code': code}
        )

        send_mail(
            subject='[健身輔助系統] 您的註冊驗證碼',
            message=f'您好，\n\n您的註冊驗證碼是 {code} 。\n此驗證碼將在 15 分鐘後失效，請盡快回到 APP 填寫。',
            from_email=settings.DEFAULT_FROM_EMAIL if hasattr(settings, 'DEFAULT_FROM_EMAIL') else 'noreply@fitness.com',
            recipient_list=[email],
            fail_silently=False,
        )

        return Response({"message": "驗證碼已寄出"}, status=status.HTTP_200_OK)


@extend_schema_view(
    post=extend_schema(summary="使用者註冊 (驗證碼)", description="前端傳入 email, password, code。驗證成功即建立帳戶。", responses={201: UserSerializer})
)
class RegisterView(generics.CreateAPIView):
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)


@extend_schema_view(
    get=extend_schema(summary="取得個人資料", description="回傳當前登入使用者的基本資料與 Profile (包含身高、體重、性別與生日)。"),
    put=extend_schema(summary="填寫/更新個人資料", description="覆寫更新使用者的身體數據。需在 Header 帶入 JWT Token。"),
    patch=extend_schema(summary="局部更新個人資料", description="部分更新使用者身體數據。需在 Header 帶入 JWT Token。")
)
class UserProfileView(generics.RetrieveUpdateAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = FillProfileSerializer

    def get_object(self):
        # We ensure a profile exists when requesting logic hits here
        # (Though signals create it automatically, this handles safe extraction for legacy users)
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile

    def retrieve(self, request, *args, **kwargs):
        # Since the user might want to see both `email` and their `profile` details in GET API
        # but the Update uses FillProfileSerializer, we can override retrieve to show everything.
        instance = request.user
        return Response(UserSerializer(instance).data)
