from rest_framework import serializers
from auth_app.models import User
from django.contrib.humanize.templatetags.humanize import naturaltime

class UserSerializer(serializers.ModelSerializer):
    full_name = serializers.SerializerMethodField()
    initials  = serializers.SerializerMethodField()

    class Meta:
        model  = User
        fields = ["id", "full_name", "initials", "email", "date_joined"]

    def get_full_name(self, obj):
        return obj.first_name or obj.email

    def get_initials(self, obj):
        name = obj.first_name or obj.email
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        return name[:2].upper()

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data['date_joined'] = naturaltime(instance.date_joined)
        return data
