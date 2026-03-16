from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, EmailVerificationToken

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['height', 'weight', 'gender', 'dob']

class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)
    username = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'profile']

    def get_username(self, obj):
        if hasattr(obj, 'profile') and obj.profile and obj.profile.nickname:
            return obj.profile.nickname
        return obj.email

# API 1: Request OTP
class SendVerificationSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)

# API 2: Complete Registration
class RegisterSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})
    code = serializers.CharField(max_length=6, required=True)

    def validate(self, attrs):
        email = attrs.get('email')
        code = attrs.get('code')

        if User.objects.filter(email=email).exists():
            raise serializers.ValidationError({"email": "這組信箱已經註冊過了。"})

        # Verify OTP
        try:
            token = EmailVerificationToken.objects.filter(email=email, code=code).latest('created_at')
            if not token.is_valid():
                raise serializers.ValidationError({"code": "驗證碼已過期，請重新發送。"})
        except EmailVerificationToken.DoesNotExist:
            raise serializers.ValidationError({"code": "驗證碼錯誤。"})

        return attrs

    def create(self, validated_data):
        # Username defaults to email
        user = User.objects.create_user(
            username=validated_data['email'],
            email=validated_data['email'],
            password=validated_data['password']
        )
        # Email is verified, so we delete tokens
        EmailVerificationToken.objects.filter(email=validated_data['email']).delete()
        return user

# API 3: Fill Profile
class FillProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='nickname', required=False, allow_blank=True)

    class Meta:
        model = UserProfile
        fields = ['height', 'weight', 'gender', 'dob', 'username']

from rest_framework_simplejwt.tokens import RefreshToken

class EmailTokenObtainSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        from django.contrib.auth import authenticate
        email = attrs.get('email')
        password = attrs.get('password')

        user = authenticate(request=self.context.get('request'), email=email, password=password)
        
        if not user or not user.is_active:
            raise serializers.ValidationError('登入失敗，信箱或密碼錯誤。', code='authorization')

        self.user = user
        refresh = RefreshToken.for_user(user)

        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }
