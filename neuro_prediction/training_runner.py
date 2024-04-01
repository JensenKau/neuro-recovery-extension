from __future__ import annotations

import numpy as np
import torch

from src.mlmodels.other_models.svm_model import SVMModel
from src.patientdata.patient_data import PatientData
from src.patientdata.patient_dataset import PatientDataset

from src.load_data import load_data

if __name__ == "__main__":
    cnn = SVMModel()
    patient_dataset = PatientDataset.load_processed_dataset("src/balanced_connectivity.pkl")
    
    cnn.initialize_model()
    
    cnn.k_fold(patient_dataset.get_dataset())
    # cnn.save_k_fold("../../trained_models/test_save2", BaseMLModel.SAVE_MODE.ALL)
    
    print(cnn.get_k_fold_performances()["avg"])
    
