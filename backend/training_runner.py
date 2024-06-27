from __future__ import annotations
import os
from pprint import pprint

from src.patientdata.patient_dataset import PatientDataset

if __name__ == "__main__":
   PatientDataset.load_processed_eeg(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train")