from __future__ import annotations
import os
from pprint import pprint

from src.dataset.transform import TransformDataset
from src.patientdata.eeg_data import PatientEEGData

if __name__ == "__main__":
    TransformDataset.delete_raw_data(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset")