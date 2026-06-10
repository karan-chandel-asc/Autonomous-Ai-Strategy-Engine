from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from Ai_strategy_engine.logger import logger


@shared_task
def send_welcome_email(email, full_name, password=None):
    logger.info(f"Sending welcome email to {email}")
    name = full_name or email.split('@')[0]

    subject = "Welcome to Nexus — Let's build your strategy"
    message = f"""Hi {name},

Welcome to Nexus! We're glad to have you on board.

Nexus is your autonomous AI strategy engine — powered by 7 specialized AI agents that work in parallel to deliver deep market analysis, competitive intelligence, risk assessments, monetization strategies, and execution roadmaps for any business objective.

Here's what you can do right now:

  1. Upload documents to your Knowledge Base — PDFs, research reports, or policy docs that ground your analysis in real context.
  2. Define your objective — describe what you're building or evaluating in plain language.
  3. Run your first analysis — all 7 agents fire in parallel and deliver a complete strategy brief in under a minute.

Your results are saved as structured reports you can revisit, compare, and build on over time.

We're excited to see what you'll build.

— The Nexus Team
"""
    send_mail(
        subject        = subject,
        message        = message,
        from_email     = settings.DEFAULT_FROM_EMAIL,
        recipient_list = [email],
        fail_silently  = False,
    )
    logger.info(f"Welcome email sent to {email}")


@shared_task
def send_password_reset_email(email, reset_url):
    logger.info(f"Sending password reset email to {email}")
    subject = "Reset your Nexus password"
    message = f"""Hi,

You requested a password reset for your Nexus account.

Click the link below to reset your password:

{reset_url}

This link is valid for one use only. If you did not request this, you can safely ignore this email.

— The Nexus Team
"""
    send_mail(
        subject        = subject,
        message        = message,
        from_email     = settings.DEFAULT_FROM_EMAIL,
        recipient_list = [email],
        fail_silently  = False,
    )
    logger.info(f"Password reset email sent to {email}")
