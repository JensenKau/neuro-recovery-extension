from __future__ import annotations
import os
import shutil
from typing import List, Tuple

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response
from django.core.files.base import ContentFile
from django.core.files import File
from django.core.files.storage import default_storage
import numpy as np
from numpy.typing import NDArray
import scipy

from ..models import PatientEEG, Patient, Prediction, AiModel
from ..serializers import ShortEEGSerializer, EEGSerializer, FCSerializer
from src.patientdata.patient_data import PatientData
from src.patientdata.eeg_data import PatientEEGData
from src.patientdata.meta_data import PatientMetaData
from src.patientdata.connectivity_data import PatientConnectivityData
from src.patientdata.data_enum import *
from src.mlmodels.pytorch_models.dynamic.cnn_dynamic2_2 import CnnDynamic2_2

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
        
        patient_eeg = None
        
        with open(os.path.join(default_storage.location, f"{folder}/{filename}.mat"), "rb") as mat_file:
            with open(os.path.join(default_storage.location, f"{folder}/{filename}.hea"), "rb") as hea_file:
                avg_fc, std_fc, static_fc = patient_data.get_fcs()
                patient_eeg = PatientEEG.objects.create(
                    name=filename,
                    patient=patient,
                    start_time=patient_data.get_start_time(),
                    end_time=patient_data.get_end_time(),
                    utility_freq=patient_data.get_utility_frequency(),
                    sampling_freq=patient_data.get_sampling_frequency(),
                    header_file=ContentFile(hea_file.read(), name=f"{filename}.hea"),
                    raw_file=ContentFile(mat_file.read(), name=f"{filename}.mat"),
                    static_fc=static_fc.tolist(),
                    avg_fc=avg_fc.tolist(),
                    std_fc=std_fc.tolist()
                )
        
        shutil.rmtree(os.path.join(default_storage.location, folder))
        
        cnn = CnnDynamic2_2()
        cnn.load_model("./mlmodel/model_0.pt")
        pred, conf = cnn.predict_result([patient_data])[0]
        
        Prediction.objects.create(
            patient_eeg=patient_eeg,
            ai_model=AiModel.objects.get(id=1),
            outcome_pred="good" if pred == 0 else "bad",
            cpc_pred=None,
            confidence=conf,
            comments=""
        )
        
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
    
    
class GetFCs(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request) -> Response:
        data = request.query_params
        
        patient_id = data["patient_id"]
        filename = data["filename"]
        
        query = PatientEEG.objects.get(patient_id=patient_id, name=filename)
        serializer = FCSerializer(query)
        
        return Response(serializer.data)
    
    
class GetEEGPoints(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()

    
    def load_patient_connectivity(self, eeg_data: PatientEEGData) -> List[List[float]]:
        actual_eeg = eeg_data.get_eeg_data()
        num_points = eeg_data.get_num_points()
        sampling_frequency = eeg_data.get_sampling_frequency()
        utility_frequency = eeg_data.get_utility_frequency()
        regions = PatientConnectivityData.BRAIN_REGION
        organized_data = [None] * len(regions)

        for i in range(len(regions)):
            if regions[i] in actual_eeg:
                organized_data[i] = actual_eeg[regions[i]]
            else:
                organized_data[i] = np.zeros(num_points)
                
        organized_data, sampling_frequency = PatientConnectivityData.preprocess_data(np.array(organized_data), sampling_frequency, utility_frequency)
        
        return organized_data.tolist()
    
    
    def get(self, request: Request) -> Response:
        data = request.query_params
        
        patient_id = data["patient_id"]
        filename = data["filename"]
        
        query = PatientEEG.objects.get(patient_id=patient_id, name=filename)

        eeg = PatientEEGData.load_eeg_data(query.header_file.path, query.raw_file.path)
        processed = self.load_patient_connectivity(eeg)
        
        return Response({
            "data": processed
        })



if __name__ == "__main__":
    pass