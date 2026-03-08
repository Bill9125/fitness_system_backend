from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from drf_spectacular.utils import extend_schema, extend_schema_view
from .serializers import RegisterSerializer, UserSerializer

@extend_schema_view(
    post=extend_schema(
        summary="使用者註冊",
        description="建立一個新帳號。會同時自動建立一個空的 UserProfile 供儲存身高、體重與性別資料。",
        responses={201: UserSerializer}
    )
)
class RegisterView(generics.CreateAPIView):
    queryset = UserSerializer.Meta.model.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer

@extend_schema_view(
    get=extend_schema(
        summary="取得個人資料",
        description="回傳當前登入使用者的基本資料與 Profile (包含身高、體重、性別與生日)。需在 Header 帶入 JWT Token。",
        responses={200: UserSerializer}
    ),
    put=extend_schema(
        summary="完整更新個人資料",
        description="覆蓋更新使用者的所有基本資料與 Profile 資料。需在 Header 帶入 JWT Token。",
        responses={200: UserSerializer}
    ),
    patch=extend_schema(
        summary="局部更新個人資料 (PATCH)",
        description="只更新有傳送的特定使用者參數，不會覆寫未傳送的欄位。需在 Header 帶入 JWT Token。",
        responses={200: UserSerializer}
    )
)
class UserProfileView(generics.RetrieveUpdateAPIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user

    def update(self, request, *args, **kwargs):
        # Update user standard fields
        user = self.get_object()
        user_data = request.data
        profile_data = user_data.pop('profile', None)

        serializer = self.get_serializer(user, data=user_data, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        # Update profile fields if provided
        if profile_data:
            from .serializers import UserProfileSerializer
            profile_serializer = UserProfileSerializer(user.profile, data=profile_data, partial=True)
            profile_serializer.is_valid(raise_exception=True)
            profile_serializer.save()

        return Response(serializer.data)
