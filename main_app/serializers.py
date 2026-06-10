from rest_framework import serializers
from .models import Thread, AgentResponse, FinalStrategy


class AgentResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model  = AgentResponse
        fields = ['agent_name', 'status', 'response', 'runtime_ms', 'created_at']


class FinalStrategySerializer(serializers.ModelSerializer):
    class Meta:
        model  = FinalStrategy
        fields = ['data', 'created_at']


class ThreadSerializer(serializers.ModelSerializer):
    agents_done  = serializers.SerializerMethodField()
    agents_total = serializers.SerializerMethodField()

    class Meta:
        model  = Thread
        fields = ['thread_id', 'objective', 'status', 'runtime_ms',
                  'agents_done', 'agents_total', 'completed_at', 'created_at']

    def get_agents_done(self, obj):
        return obj.agent_responses.filter(status='complete').count()

    def get_agents_total(self, obj):
        return 7


class ReportSerializer(serializers.ModelSerializer):
    agent_responses = AgentResponseSerializer(many=True, read_only=True)
    final_strategy  = FinalStrategySerializer(read_only=True)

    class Meta:
        model  = Thread
        fields = [
            'thread_id', 'objective', 'status',
            'runtime_ms', 'failed_reason', 'completed_at', 'created_at',
            'agent_responses', 'final_strategy',
        ]
