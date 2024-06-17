from __future__ import annotations
from patientdata.eeg_data import PatientEEGData

if __name__ == "__main__":
    eeg = PatientEEGData.load_eeg_data(
        header_file=r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train\1016\1016_006_012_EEG.hea",
        content_file=r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train\1016\1016_006_012_EEG.mat"
    )
    
    print(eeg.get_eeg_data())