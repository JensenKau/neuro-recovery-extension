from __future__ import annotations

import numpy as np
import random
import optuna
import logging
import sys

from mlmodels.cnn_simple import CnnSimple
from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

if __name__ == "__main__":
    cnn = CnnSimple()
    patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")

    cnn.initialize_model()
    
    res = cnn.k_fold(patient_dataset.get_dataset())
    
    print(res)
    
    # res = cnn.predict_result(patient_dataset.get_dataset())

    # print(res)
    # print(len(res))