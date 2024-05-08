from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response
from django.cores.files import File

from ..models import PatientEEG

class GenerateEEGData(CreateAPIView):    
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        data = request.data
        
        patient_id = data["patient_id"]
        heaFile = data["heaFile"]
        matFile = data["matFile"]
        
        # filecount = len(PatientEEG.objects.filter(patient_id=patient_id))
        
        # with open(f"./tmp/{patient_id}_{filecount}.txt", "wb") as file:
        #     for chunk in heaFile.chunks():
        #         file.write(chunk)
                
        # with open(f"./tmp/{patient_id}_{filecount}.mat", "wb") as file:
        #     for chunk in matFile.chunks():
        #         file.write(chunk)
                
        
        
        return Response({})


if __name__ == "__main__":
    pass