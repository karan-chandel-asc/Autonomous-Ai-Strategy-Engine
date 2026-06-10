from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import login, logout
from pydantic import ValidationError

from .serializers import UserSerializer
from .schemas import success_response, error_response, LoginSchema, SignupSchema, ForgotPasswordSchema, ResetPasswordSchema
from .services import _service
from .tasks import send_password_reset_email
from Ai_strategy_engine.logger import logger


class UserSignupView(APIView):
    authentication_classes = []
    permission_classes     = []

    def get(self, request):
        logger.info("GET Request received for signup page")
        from django.shortcuts import render ,redirect
        if request.user.is_authenticated:
            return redirect('/main-app/dashboard/')
        return render(request, 'ase_login.html')

    def post(self, request):
        logger.info("Request comes for user signup")
        try:
            data = request.data.dict() if hasattr(request.data, 'dict') else dict(request.data)
            try:
                schema = SignupSchema(**data)
            except ValidationError as e:
                err     = e.errors()[0]
                message = err['msg'].replace('Value error, ', '')
                return Response(error_response(message=message), status=status.HTTP_400_BAD_REQUEST)

            # email already exists
            if _service.is_user_exists(schema.email):
                return Response(error_response(message="User with this email already exists"), status=status.HTTP_400_BAD_REQUEST)

            user, error_message = _service.create_user(schema.model_dump())
            if user is None:
                return Response(error_response(message=error_message), status=status.HTTP_400_BAD_REQUEST)

            return Response(
                success_response(message="User created successfully", data=UserSerializer(user).data),
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LoginView(APIView):
    authentication_classes = []
    permission_classes     = []

    def get(self, request):
        logger.info("GET Request received for login page")
        from django.shortcuts import render,redirect
        if request.user.is_authenticated:
            return redirect('/main-app/dashboard/')
        return render(request, 'ase_login.html')

    def post(self, request):
        logger.info("Request comes for user login")
        try:
            data = request.data.dict() if hasattr(request.data, 'dict') else dict(request.data)
            try:
                schema = LoginSchema(**data)
            except ValidationError as e:
                err     = e.errors()[0]
                message = err['msg'].replace('Value error, ', '')
                return Response(error_response(message=message), status=status.HTTP_400_BAD_REQUEST)

            user, error_message = _service.get_user_by_email(schema.email, schema.password)
            if user is None:
                return Response(error_response(message=error_message), status=status.HTTP_401_UNAUTHORIZED)

            user.backend = 'django.contrib.auth.backends.ModelBackend'
            login(request, user)
            refresh = RefreshToken.for_user(user)
            return Response(
                success_response(
                    message="Login successful",
                    data={
                        "user":    UserSerializer(user).data,
                        "access":  str(refresh.access_token),
                        "refresh": str(refresh),
                    }
                ),
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LogoutView(APIView):
    authentication_classes = []
    permission_classes     = []

    def post(self, request):
        logger.info("Request comes for user logout")
        logout(request)
        return Response(success_response(message="Logged out successfully"), status=status.HTTP_200_OK)


class UserProfileView(APIView):
    permission_classes     = [IsAuthenticated]

    def get(self, request):
        logger.info("Request comes for user profile")
        try:
            return Response(
                success_response(message="Profile fetched successfully", data=UserSerializer(request.user).data),
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def put(self, request):
        logger.info("Request comes for updating user profile")
        try:
            user = request.user
            data = request.data.dict() if hasattr(request.data, 'dict') else dict(request.data)

            full_name = data.get("full_name", "").strip()
            current_password = data.get("current_password", "").strip()
            new_password     = data.get("new_password", "").strip()

            if full_name:
                user.first_name = full_name

            if current_password or new_password:
                if not current_password:
                    return Response(error_response(message="Current password is required to set a new password"), status=status.HTTP_400_BAD_REQUEST)
                if not new_password:
                    return Response(error_response(message="New password cannot be empty"), status=status.HTTP_400_BAD_REQUEST)
                if not user.check_password(current_password):
                    return Response(error_response(message="Current password is incorrect"), status=status.HTTP_400_BAD_REQUEST)
                if len(new_password) < 6:
                    return Response(error_response(message="New password must be at least 6 characters"), status=status.HTTP_400_BAD_REQUEST)
                user.set_password(new_password)

            user.save()
            return Response(
                success_response(message="Profile updated successfully", data=UserSerializer(user).data),
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OAuthCallbackView(APIView):
    authentication_classes = []
    permission_classes     = []

    def get(self, request):
        logger.info("OAuth callback received")
        from django.shortcuts import render
        return render(request, 'oauth_callback.html')


class ForgotPasswordView(APIView):
    authentication_classes = []
    permission_classes     = []

    def get(self, request):
        logger.info("GET Request received for forgot password page")
        from django.shortcuts import render
        return render(request, 'ase_forgot_password.html')

    def post(self, request):
        logger.info("Request comes for forgot password")
        try:
            data = request.data.dict() if hasattr(request.data, 'dict') else dict(request.data)
            try:
                schema = ForgotPasswordSchema(**data)
            except ValidationError as e:
                err     = e.errors()[0]
                message = err['msg'].replace('Value error, ', '')
                return Response(error_response(message=message), status=status.HTTP_400_BAD_REQUEST)

            uid, token = _service.generate_reset_token(schema.email)

            if uid and token:
                reset_url = f"{request.scheme}://{request.get_host()}/auth-api/reset_password_page/?uid={uid}&token={token}"
                send_password_reset_email.delay(schema.email, reset_url)

            return Response(
                success_response(message="If this email is registered, a reset link has been sent."),
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ResetPasswordPageView(APIView):
    authentication_classes = []
    permission_classes     = []

    def get(self, request):
        logger.info("GET Request received for reset password page")
        from django.shortcuts import render
        return render(request, 'ase_reset_password.html')


class ResetPasswordView(APIView):
    authentication_classes = []
    permission_classes     = []

    def post(self, request):
        logger.info("Request comes for reset password")
        try:
            data = request.data.dict() if hasattr(request.data, 'dict') else dict(request.data)
            try:
                schema = ResetPasswordSchema(**data)
            except ValidationError as e:
                err     = e.errors()[0]
                message = err['msg'].replace('Value error, ', '')
                return Response(error_response(message=message), status=status.HTTP_400_BAD_REQUEST)

            user, error_message = _service.reset_password(schema.uid, schema.token, schema.password)
            if user is None:
                return Response(error_response(message=error_message), status=status.HTTP_400_BAD_REQUEST)

            return Response(
                success_response(message="Password reset successfully. Please sign in."),
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(error_response(message=str(e)), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
