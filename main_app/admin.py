from django.contrib import admin
from .models import Thread, Document, Vector_storage, Chunk, AgentResponse, FinalStrategy ,KnowledgeBase


@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display    = ('thread_id', 'owner', 'status', 'runtime_ms', 'created_at', 'completed_at')
    list_filter     = ('status',)
    search_fields   = ('thread_id', 'objective')
    readonly_fields = ('thread_id', 'created_at', 'completed_at')


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display    = ('id', 'thread', 'document_path', 'document_hash', 'created_at')
    search_fields   = ('document_hash',)
    readonly_fields = ('created_at',)


@admin.register(Vector_storage)
class VectorStorageAdmin(admin.ModelAdmin):
    list_display  = ('id', 'thread', 'vector_path')
    search_fields = ('vector_path',)


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display  = ('id', 'thread', 'chunk_hash')
    search_fields = ('chunk_hash', 'content')


@admin.register(AgentResponse)
class AgentResponseAdmin(admin.ModelAdmin):
    list_display    = ('id', 'thread', 'agent_name', 'status', 'runtime_ms', 'created_at')
    list_filter     = ('agent_name', 'status')
    search_fields   = ('thread__thread_id',)
    readonly_fields = ('created_at',)


@admin.register(FinalStrategy)
class FinalStrategyAdmin(admin.ModelAdmin):
    list_display    = ('id', 'thread', 'created_at')
    search_fields   = ('thread__thread_id',)
    readonly_fields = ('created_at',)


admin.site.register(KnowledgeBase)