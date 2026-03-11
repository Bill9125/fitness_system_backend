from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model

UserModel = get_user_model()

class EmailBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        # Allow checking either 'username' (which DRF TokenObtainPairView passes by default)
        # or exactly 'email' kwarg
        raw_email = kwargs.get('email', username)
        if raw_email is None:
            return None
        
        try:
            user = UserModel.objects.get(email=raw_email)
        except UserModel.DoesNotExist:
            return None

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
