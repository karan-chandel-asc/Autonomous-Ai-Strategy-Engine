import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Ai_strategy_engine.settings')
# macOS: Objective-C runtime initializes in threads; fork() during that window
# causes SIGABRT. This flag disables the crash-on-unsafe-fork check.
os.environ.setdefault('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
# Prevents HuggingFace tokenizer semaphore leaks inside Celery workers.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

app = Celery('Ai_strategy_engine')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
