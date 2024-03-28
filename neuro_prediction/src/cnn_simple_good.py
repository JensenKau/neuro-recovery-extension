from __future__ import annotations

import numpy as np
import torch

from mlmodels.base_mlmodel import BaseMLModel
from mlmodels.cnn_simple_static import CnnSimpleStatic
from mlmodels.cnn_simple_static2 import CnnSimpleStatic2
from mlmodels.cnn_simple import CnnSimple
from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

if __name__ == "__main__":
    cnn = CnnSimpleStatic2()
    patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")

    # cnn.initialize_model()
    
    # cnn.k_fold(patient_dataset.get_dataset())
    # cnn.save_k_fold("../../trained_models/test_save2", BaseMLModel.SAVE_MODE.ALL)
    
    # print(cnn.get_k_fold_performances()["avg"])
    
    # res = cnn.predict_result(patient_dataset.get_dataset())

    # print(res)
    # print(len(res))
    
    cnn.tune_paramters(100, patient_dataset.get_dataset())
    
    # print(torch.cuda.get_device_name())
    # print(torch.cuda.device())
    
    # cnn.load_model("../../trained_models/test_save2/model_1.pt")