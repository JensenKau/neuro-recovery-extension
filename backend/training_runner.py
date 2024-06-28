from __future__ import annotations
import os
from pprint import pprint

from src.patientdata.patient_dataset import PatientDataset
from src.dataset.transform import TransformDataset
from src.patientdata.eeg_data import PatientEEGData
from src.patientdata.data_enum import PatientOutcome
from src.evaluator.kfold import KFold

if __name__ == "__main__":
   # dataset = PatientDataset.load_processed_eeg(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train_ftsurrogate")
   # dataset.save_dataset("dataset_ft.pkl")
   
   dataset = PatientDataset.load_dataset("dataset_ft.pkl").get_dataset()
   
   KFold().split_data(dataset)