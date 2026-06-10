from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str

from auth_app.models import User
from auth_app.tasks import send_welcome_email


class AuthenticationService:

    def is_user_exists(self, email):
        return User.objects.filter(email=email).exists()

    def create_user(self, data):
        email     = data.get("email")
        full_name = data.get("full_name")
        password  = data.get("password")

        if self.is_user_exists(email):
            return None, "User with this email already exists"

        user = User.objects.create_user(
            email      = email,
            password   = password,
            first_name = full_name,
        )
        send_welcome_email.delay(email, full_name)
        return user, None

    def get_user_by_email(self, email, password):
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return None, "Invalid email or password"

        if not user.check_password(password):
            return None, "Invalid email or password"

        return user, None

    def generate_reset_token(self, email):
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return None, None

        token = PasswordResetTokenGenerator().make_token(user)
        uid   = urlsafe_base64_encode(force_bytes(user.pk))
        return uid, token

    def reset_password(self, uid, token, new_password):
        try:
            user_id = force_str(urlsafe_base64_decode(uid))
            user    = User.objects.get(pk=user_id)
        except Exception:
            return None, "Invalid reset link"

        if not PasswordResetTokenGenerator().check_token(user, token):
            return None, "Reset link is invalid or has expired"

        user.set_password(new_password)
        user.save()
        return user, None


_service = AuthenticationService()
