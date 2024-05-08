from __future__ import annotations

from rest_framework.serializers import ModelSerializer, ListField, FileField, Serializer

from ..models import PatientEEG

class ShortEEGSerializer(ModelSerializer):
    class Meta:
        model = PatientEEG
        fields = ["patient", "name", "created_at"]