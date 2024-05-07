from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser

from ..serializers import MultipleFileSerializer


class GenerateEEGData(CreateAPIView):    
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        data = request.data
        
        heaFile = data["heaFile"]
        matFile = data["matFile"]
        
        counter = 0
        
        for chunk in heaFile.chunks():
            counter += 1
            
        print(counter)
        
        return Response({})


if __name__ == "__main__":
    pass