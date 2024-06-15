from __future__ import annotations

from rest_framework.serializers import ModelSerializer

from ..models import User


class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ["email", "fullname", "role"]


if __name__ == "__main__":
    pass