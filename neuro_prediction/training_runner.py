from __future__ import annotations
import random

import numpy as np
import torch
from nilearn import plotting
import matplotlib.pyplot as plt
from PIL import Image

from src.patientdata.patient_data import PatientData
from src.patientdata.patient_dataset import PatientDataset
from src.patientdata.connectivity_data import PatientConnectivityData
from src.mlmodels.pytorch_models.dynamic.cnn_dynamic2_2 import CnnDynamic2_2

if __name__ == "__main__":
    heaFile = "/root/neuro-recovery-prediction/data/training/training/1016/1016_006_012_EEG.hea"
    matFile = "/root/neuro-recovery-prediction/data/training/training/1016/1016_006_012_EEG.mat"
    metaFile = "/root/neuro-recovery-prediction/data/training/training/1016/1016.txt"
    
    data = PatientData.load_patient_data(metaFile, heaFile, matFile)
    
    # # Input random data
    # random.seed(4567)
    # data.connectivity.static_fc = [[random.random() for _ in range(22)] for _ in range(22)]
    # data.connectivity.avg_fc = [[random.random() for _ in range(22)] for _ in range(22)]
    # data.connectivity.std_fc = [[random.random() for _ in range(22)] for _ in range(22)]
    
    dataset = [data]
    
    model = CnnDynamic2_2()
    model.load_model("/root/neuro-recovery-prediction/neuro_prediction/mlmodel/model_0.pt")
    
    res = model.predict_result(dataset)[0]
    
    print(f"Prediction: {res[0]}, Confidence: {res[1]}")
    
    