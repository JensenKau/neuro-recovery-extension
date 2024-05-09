from __future__ import annotations

from rest_framework.serializers import ModelSerializer, ListField, FileField, Serializer

from ..models import PatientEEG

class ShortEEGSerializer(ModelSerializer):
    class Meta:
        model = PatientEEG
        fields = ["patient", "name", "created_at"]
        

class EEGSerializer(ModelSerializer):
    class Meta:
        model = PatientEEG
        exclude = ["raw_file", "static_fc", "avg_fc", "std_fc"]
        

if __name__ == "__main__":
    pass