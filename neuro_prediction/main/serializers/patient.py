from __future__ import annotations

from rest_framework.serializers import ModelSerializer

from ..models import Patient


class ShortPatientSerializer(ModelSerializer):
    class Meta:
        model = Patient
        fields = ["id", "name"]


class PatientSerializer(ModelSerializer):
    class Meta:
        model = Patient
        fields = "__all__"


if __name__ == "__main__":
    pass