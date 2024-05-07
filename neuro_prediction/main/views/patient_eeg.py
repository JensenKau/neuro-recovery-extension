from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from ..serializers import MultipleFileSerializer


class GenerateEEGData(CreateAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        serializer = MultipleFileSerializer(data=request.data or None)
        serializer.is_valid(raise_exception=True)
        files = serializer.validated_data.get("files")
        
        for file in files:
            print(file)
        
        return Response({})


if __name__ == "__main__":
    pass