import uuid
from django.db import models


class Thread(models.Model):
    STATUS_CHOICES = [
        ('queued',   'Queued'),
        ('running',  'Running'),
        ('complete', 'Complete'),
        ('failed',   'Failed'),
    ]

    owner        = models.ForeignKey("auth_app.User", on_delete=models.CASCADE, related_name="threads")
    thread_id    = models.CharField(max_length=255, primary_key=True)
    objective    = models.TextField(blank=True, default="")
    status       = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    runtime_ms   = models.PositiveIntegerField(null=True, blank=True)
    failed_reason= models.CharField(blank=True,null= True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Thread {self.thread_id} [{self.status}]"


class Document(models.Model):
    thread        = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="documents")
    document_path = models.FileField(upload_to="documents/")
    document_hash = models.CharField(max_length=64)
    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.document_path.name


class Vector_storage(models.Model):
    thread      = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="vector_storage")
    vector_path = models.CharField(max_length=500)
    metadata    = models.JSONField()

    def __str__(self):
        return f"Vectors for {self.thread_id}"


class Chunk(models.Model):
    thread     = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="chunks")
    chunk_hash = models.CharField(max_length=64)
    content    = models.TextField()

    def __str__(self):
        return f"Chunk {self.chunk_hash[:8]}"


class AgentResponse(models.Model):
    AGENT_CHOICES = [
        ('executive_summary',       'Executive Summary'),
        ('market_analysis',         'Market Analysis'),
        ('competitive_landscape',   'Competitive Landscape'),
        ('monetization_strategy',   'Monetization Strategy'),
        ('risk_assessment',         'Risk Assessment'),
        ('product_roadmap',         'Product Roadmap'),
        ('weakness_review',         'Weakness Review'),
    ]

    STATUS_CHOICES = [
        ('pending',  'Pending'),
        ('running',  'Running'),
        ('complete', 'Complete'),
        ('failed',   'Failed'),
    ]

    thread     = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="agent_responses")
    agent_name = models.CharField(max_length=50, choices=AGENT_CHOICES)
    status     = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    response   = models.JSONField(null=True, blank=True)
    runtime_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        unique_together = ('thread', 'agent_name')

    def __str__(self):
        return f"{self.agent_name} [{self.status}] — {self.thread_id}"


class FinalStrategy(models.Model):
    thread     = models.OneToOneField(Thread, on_delete=models.CASCADE, related_name="final_strategy")
    data       = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Strategy for {self.thread_id}"


class KnowledgeBase(models.Model):
    STATUS_CHOICES = [
        ('pending',  'Pending'),
        ('indexing', 'Indexing'),
        ('indexed',  'Indexed'),
        ('failed',   'Failed'),
    ]

    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner       = models.ForeignKey("auth_app.User", on_delete=models.CASCADE, related_name="knowledge_bases")
    name        = models.CharField(max_length=255)
    file        = models.FileField(upload_to="knowledge_base/")
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    chunk_count = models.PositiveIntegerField(default=0)
    error       = models.TextField(blank=True, default="")
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} [{self.status}]"
