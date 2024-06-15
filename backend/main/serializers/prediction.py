from __future__ import annotations

from rest_framework.serializers import ModelSerializer

from ..models import Prediction
from .ai_model import ShortAiModelSerializer

class PredictionSerializer(ModelSerializer):
    ai_model = ShortAiModelSerializer(read_only=True)
    
    class Meta:
        model = Prediction
        exclude = ["cpc_pred", "created_at", "updated_at"]


if __name__ == "__main__":
    pass