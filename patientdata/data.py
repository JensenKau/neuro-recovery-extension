from __future__ import annotations
from .eeg_data import PatientEEGData
from .meta_data import PatientMetaData
from .connectivity_data import PatientConnectivityData

class PatientData:
    def __init__(self, eeg: PatientEEGData, meta: PatientMetaData) -> None:
        self.eeg = eeg
        self.meta = meta
        self.connectivity = PatientConnectivityData.load_patient_connectivity(eeg)

if __name__ == "__main__":
    pass