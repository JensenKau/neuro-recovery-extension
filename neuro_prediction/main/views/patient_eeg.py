from __future__ import annotations
import os
import shutil

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response
from django.core.files.base import ContentFile
from django.core.files import File
from django.core.files.storage import default_storage

from ..models import PatientEEG, Patient
from ..serializers import ShortEEGSerializer, EEGSerializer
from src.patientdata.patient_data import PatientData
from src.patientdata.eeg_data import PatientEEGData
from src.patientdata.meta_data import PatientMetaData
from src.patientdata.data_enum import *

class GenerateEEGData(CreateAPIView):    
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request) -> Response:
        data = request.data
        
        patient_id = int(data["patient_id"])
        filename = data["filename"]
        heaFile = data["heaFile"]
        matFile = data["matFile"]
        
        patient = Patient.objects.get(id=patient_id)
        
        folder = f"raw_eeg/{patient_id}"
        default_storage.save(f"{folder}/{filename}.hea", heaFile)
        default_storage.save(f"{folder}/{filename}.mat", matFile)
        
        eeg_data = PatientEEGData.load_eeg_data(
            os.path.join(default_storage.location, f"{folder}/{filename}.hea"),
            os.path.join(default_storage.location, f"{folder}/{filename}.mat")
        )
        
        patient_data = PatientData(
            eeg=eeg_data,
            meta=PatientMetaData(
                patient_id=patient.id,
                hospital=None,
                age=patient.age,
                sex= PatientSex.NONE if patient.sex is None else PatientSex.MALE if patient.sex == "male" else PatientSex.FEMALE,
                rosc=patient.rosc,
                ohca=patient.ohca,
                shockable_rhythm=patient.shockable_rhythm,
                ttm=patient.ttm,
                outcome=PatientOutcome.NONE,
                cpc=0
            )
        )
        
        static_fc, avg_fc, std_fc = patient_data.get_fcs()
        
        with open(os.path.join(default_storage.location, f"{folder}/{filename}.mat"), "rb") as file:
            PatientEEG.objects.create(
                name=filename,
                patient=patient,
                start_time=patient_data.get_start_time(),
                end_time=patient_data.get_end_time(),
                utility_freq=patient_data.get_utility_frequency(),
                sampling_freq=patient_data.get_sampling_frequency(),
                raw_file=ContentFile(file.read(), name=f"{filename}.mat"),
                static_fc=static_fc.tolist(),
                avg_fc=avg_fc.tolist(),
                std_fc=std_fc.tolist()
            )
        
        shutil.rmtree(os.path.join(default_storage.location, folder))
                
        return Response({})
    
    
class GetEEGs(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request) -> Response:
        data = request.query_params
        
        patient_id = data["patient_id"]
        
        queries = PatientEEG.objects.filter(patient_id=patient_id)
        serializer = ShortEEGSerializer(queries, many=True)
        
        return Response({"eegs": serializer.data})
    

class GetEEG(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request) -> Response:
        data = request.query_params
        
        patient_id = data["patient_id"]
        filename = data["filename"]
        
        query = PatientEEG.objects.get(patient_id=patient_id, name=filename)
        serializer = EEGSerializer(query)
        
        return Response(serializer.data)
    


if __name__ == "__main__":
    pass