from __future__ import annotations

import numpy as np
import torch

from mlmodels.base_mlmodel import BaseMLModel
from mlmodels.cnn_simple_static import CnnSimpleStatic
from mlmodels.cnn_simple import CnnSimple
from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

if __name__ == "__main__":
    # cnn = CnnSimpleStatic()
    # patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")

    # cnn.initialize_model()
    # cnn.set_save_k_fold(BaseMLModel.SAVE_MODE.ALL, "../../trained_models/test_save2")
    
    # res = cnn.k_fold(patient_dataset.get_dataset())
    
    # print(res)
    
    # res = cnn.predict_result(patient_dataset.get_dataset())

    # print(res)
    # print(len(res))
    
    # cnn.tune_paramters(50, patient_dataset.get_dataset())
    
    # print(torch.cuda.get_device_name())
    # print(torch.cuda.device())
    
    print(type(float(np.mean([1, 2, 3], axis=0))))
    print("")