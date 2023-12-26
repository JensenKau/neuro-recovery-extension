from __future__ import annotations

from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

if __name__ == "__main__":
    patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")
    
    