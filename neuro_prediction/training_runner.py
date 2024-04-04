from __future__ import annotations

import numpy as np
import torch

from src.mlmodels.other_models.svm_model import SVMModel
from src.mlmodels.pytorch_models.static.cnn_simple_static2_3 import CnnSimpleStatic2_3
from src.mlmodels.pytorch_models.static.cnn_static_flex5 import CnnStaticFlex5
from src.mlmodels.pytorch_models.static.cnn_static_flex5_2 import CnnStaticFlex5_2
from src.mlmodels.pytorch_models.static.cnn_static3 import CnnStatic3
from src.mlmodels.pytorch_models.static.cnn_static_flex3_dense import CnnStaticFlex3_Dense
from src.mlmodels.pytorch_models.dynamic.cnn_dynamic2_2 import CnnDynamic2_2
from src.mlmodels.pytorch_models.dynamic.cnn_dynamic2_3 import CnnDynamic2_3
from src.mlmodels.pytorch_models.dynamic.cnn_dynamic2_4 import CnnDynamic2_4
from src.mlmodels.pytorch_models.hybrid.cnn_hybrid import CnnHybrid
from src.mlmodels.pytorch_models.hybrid.cnn_hybrid2 import CnnHybrid2
from src.patientdata.patient_data import PatientData
from src.patientdata.patient_dataset import PatientDataset

from src.load_data import load_data

if __name__ == "__main__":
    cnn = CnnHybrid2()
    patient_dataset = PatientDataset.load_processed_dataset("src/balanced_connectivity.pkl")
    
    # cnn.k_fold(patient_dataset.get_dataset())
    # # cnn.save_k_fold("../../trained_models/test_save2", BaseMLModel.SAVE_MODE.ALL)
    
    # print(cnn.get_k_fold_performances()["avg"])
    
    cnn.tune_paramters(500, patient_dataset.get_dataset())
    cnn.clear_tmp_folder()
    
    # ten1 = torch.randn((1, 2, 4, 4))
    # ten2 = torch.randn((1, 3, 2, 2))
    
    # print(torch.cat((ten1, ten2), 1).shape)