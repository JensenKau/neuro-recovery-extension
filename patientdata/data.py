from __future__ import annotations
from .eeg_data import PatientEEGData
from .meta_data import PatientMetaData

class PatientData:
    def __init__(self, eeg: PatientEEGData, meta: PatientMetaData) -> None:
        self.eeg = eeg
        self.meta = meta

if __name__ == "__main__":
    pass