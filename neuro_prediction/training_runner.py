from __future__ import annotations

import numpy as np
import torch

from src.mlmodels.pytorch_models.static.cnn_simple_static import CnnSimpleStatic
from src.mlmodels.pytorch_models.static.cnn_simple_static2 import CnnSimpleStatic2
from src.mlmodels.pytorch_models.static.cnn_simple_static2_2 import CnnSimpleStatic2_2
from src.mlmodels.pytorch_models.static.cnn_simple_static2_3 import CnnSimpleStatic2_3
from src.mlmodels.pytorch_models.hybrid.cnn_hybrid_2 import CnnHybrid1_2
from src.mlmodels.pytorch_models.hybrid.cnn_hybrid2 import CnnHybrid2
from src.patientdata.patient_data import PatientData
from src.patientdata.patient_dataset import PatientDataset

from src.load_data import load_data

if __name__ == "__main__":
    cnn = CnnSimpleStatic2_2()
    patient_dataset = PatientDataset.load_processed_dataset("src/balanced_connectivity.pkl")
    
    # cnn.load_model("/root/neuro-recovery-prediction/trained_models/pytorch_cnn_simple_static_2_2/model_0.pt")
    # res = cnn.predict_result(patient_dataset.get_dataset())
    
    # for i in range(len(res)):
    #     print(f"Input {i}: Result {res[i][0]}, Probability {res[i][1]}")
        
    # cnn.initialize_model()
    # cnn.k_fold(patient_dataset.get_dataset())
    # # cnn.save_k_fold("../../trained_models/test_save2", BaseMLModel.SAVE_MODE.ALL)
    
    # print(cnn.get_k_fold_performances()["avg"])
    
    # cnn.tune_paramters(1000, patient_dataset.get_dataset())
    # cnn.clear_tmp_folder()