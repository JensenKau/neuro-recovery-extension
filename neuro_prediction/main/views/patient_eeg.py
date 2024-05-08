from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from ..models import PatientEEG

class GenerateEEGData(CreateAPIView):    
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        data = request.data
        
        patient_id = int(data["patient_id"])
        heaFile = data["heaFile"]
        matFile = data["matFile"]
        
        filecount = len(PatientEEG.objects.filter(patient_id=patient_id))
        
        default_storage.save(f"header_eeg/{patient_id}_{filecount}.hea", heaFile)
        default_storage.save(f"raw_eeg/{patient_id}_{filecount}.mat", matFile)
        
        # with open(f"./tmp/{patient_id}_{filecount}.txt", "wb") as file:
        #     for chunk in heaFile.chunks():
        #         file.write(chunk)
                
        # with open(f"./tmp/{patient_id}_{filecount}.mat", "wb") as file:
        #     for chunk in matFile.chunks():
        #         file.write(chunk)
        
        return Response({})


if __name__ == "__main__":
    pass