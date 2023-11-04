from __future__ import annotations

from .eeg_data import PatientEEGData

class PatientConnectivityData:
    BRAIN_REGION = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", "Oz", "F9"]
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def load_patient_connectivity(eeg_data: PatientEEGData) -> PatientConnectivityData:
        actual_eeg = eeg_data.get_eeg_data()
        
    
if __name__ == "__main__":
    pass