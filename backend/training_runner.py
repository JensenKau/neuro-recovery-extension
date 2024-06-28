from __future__ import annotations
import os
from pprint import pprint

from src.patientdata.patient_dataset import PatientDataset
from src.dataset.transform import TransformDataset
from src.patientdata.eeg_data import PatientEEGData

if __name__ == "__main__":
   TransformDataset.apply_ft_surrogate(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train_ftsurrogate")