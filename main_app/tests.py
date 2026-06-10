from django.test import TestCase

# Create your tests here.
from google import genai

client = genai.Client()
models = client.models.list()
for m in models:
    print(m.name)