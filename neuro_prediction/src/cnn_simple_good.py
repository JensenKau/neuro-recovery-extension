from __future__ import annotations

import numpy as np
import random
import optuna
import logging
import sys

from mlmodels.base_mlmodel import BaseMLModel
from mlmodels.cnn_simple_static import CnnSimpleStatic
from mlmodels.cnn_simple import CnnSimple
from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

if __name__ == "__main__":
    cnn = SVMModel()
    patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")

    cnn.initialize_model()
    # cnn.set_save_k_fold(BaseMLModel.SAVE_MODE.BEST, "../../test_save")
    
    # res = cnn.k_fold(patient_dataset.get_dataset())
    
    # print(res)
    
    # res = cnn.predict_result(patient_dataset.get_dataset())

    # print(res)
    # print(len(res))
    
    cnn.tune_paramters(50, patient_dataset.get_dataset())