from django.http import HttpResponseRedirect
from django.contrib.auth import login
from rest_framework_simplejwt.tokens import RefreshToken

from auth_app.models import User
from auth_app.tasks import send_welcome_email


def get_or_create_user(backend, details, response, user=None, *args, **kwargs):
    if user:
        return {'user': user}

    email = details.get('email')

    # GitHub fallback — private emails come in response emails list
    if not email and hasattr(response, '__iter__'):
        emails = backend.get_user_details(response).get('email')
        email  = emails

    if not email:
        uid   = kwargs.get('uid', '')
        email = f"{backend.name}_{uid}@oauth.placeholder"

    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        full_name = details.get('fullname') or details.get('first_name', '')
        user = User.objects.create_user(
            email      = email,
            password   = email,
            first_name = full_name,
        )
        send_welcome_email.delay(email, full_name)

    return {'user': user}


def generate_jwt_and_redirect(backend, user, *args, **kwargs):
    if not user:
        return HttpResponseRedirect('/auth-api/user_signup/')

    user.backend = 'django.contrib.auth.backends.ModelBackend'
    login(backend.strategy.request, user)

    refresh = RefreshToken.for_user(user)
    access  = str(refresh.access_token)
    refresh_token = str(refresh)

    return HttpResponseRedirect(
        f'/auth-api/oauth_callback/?access={access}&refresh={refresh_token}'
    )
