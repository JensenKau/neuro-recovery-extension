from __future__ import annotations

from rest_framework.serializers import ModelSerializer, ListField, FileField, Serializer

class MultipleFileSerializer(Serializer):
    files = ListField(child=FileField)

if __name__ == "__main__":
    pass