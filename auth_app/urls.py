from django.urls import path
from .views import UserSignupView, LoginView, LogoutView, UserProfileView, ForgotPasswordView, ResetPasswordPageView, ResetPasswordView, OAuthCallbackView

urlpatterns = [
    path("user_signup/",          UserSignupView.as_view(),      name="user_signup"),
    path("user_login/",           LoginView.as_view(),            name="user_login"),
    path("user_logout/",          LogoutView.as_view(),           name="user_logout"),
    path("user_profile/",         UserProfileView.as_view(),      name="user_profile"),
    path("forgot_password/",      ForgotPasswordView.as_view(),   name="forgot_password"),
    path("reset_password_page/",  ResetPasswordPageView.as_view(),name="reset_password_page"),
    path("reset_password/",       ResetPasswordView.as_view(),    name="reset_password"),
    path("oauth_callback/",       OAuthCallbackView.as_view(),    name="oauth_callback"),
]
