from __future__ import annotations

from rest_framework.serializers import ModelSerializer

from ..models import AiModel


class ShortAiModelSerializer(ModelSerializer):
    class Meta:
        model = AiModel
        fields = ["id", "name"]


if __name__ == "__main__":
    pass