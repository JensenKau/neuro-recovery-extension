from __future__ import annotations
import warnings

from numpy.typing import NDArray
import numpy as np
from nilearn.connectome import ConnectivityMeasure

from src.patientdata.eeg_data import PatientEEGData


class PatientConnectivityData:
    DYNAMIC_TIME_SIZE = 60
    DYNAMIC_SHIFT_SIZE = 0.1
    
    BRAIN_REGION = [
        "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", 
        "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", 
        "Oz", "F9"
    ]
    
    def __init__(self, dynamic_fc: NDArray, static_fc: NDArray) -> None:
        self.dynamic_fc = dynamic_fc
        self.static_fc = static_fc
        self.avg_fc = np.mean(dynamic_fc, axis=0)
        self.std_fc = np.std(dynamic_fc, axis=0)
    
    
    @classmethod
    def load_patient_connectivity(cls, eeg_data: PatientEEGData) -> PatientConnectivityData:
        sampling_frequency = eeg_data.get_sampling_frequency()
        table = eeg_data.convert_eeg_to_table()
        window_size = cls.DYNAMIC_TIME_SIZE * sampling_frequency
        shift_size = int(window_size * cls.DYNAMIC_SHIFT_SIZE)
        index = 0
        static_fc = None
        dynamic_fc = []
        measure = ConnectivityMeasure(kind="correlation")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            static_fc = measure.fit_transform(np.array([table]).swapaxes(1, 2))[0]
            while index < len(table[0]):
                start = index
                end = int(min(start + window_size, len(table[0])))
                current_fc = measure.fit_transform(np.array([table[:, start:end]]).swapaxes(1, 2))[0]
                
                if not np.isnan(current_fc).any():
                    dynamic_fc.append(current_fc)
                else:
                    dynamic_fc.append(np.zeros((len(cls.BRAIN_REGION), len(cls.BRAIN_REGION)), dtype=np.float64))
                    
                index += shift_size

        return PatientConnectivityData(np.asarray(dynamic_fc, dtype=np.float64), static_fc)
    
    
    def get_avg_fc(self) -> NDArray:
        return self.avg_fc
    
    
    def get_std_fc(self) -> NDArray:
        return self.std_fc

    
    def get_static_fc(self) -> NDArray:
        return self.static_fc
    
    
    def get_dynamic_fc(self) -> NDArray:
        return self.dynamic_fc
    
    
    def delete_dynamic_fc(self) -> None:
        self.dynamic_fc = None
    
    
if __name__ == "__main__":
    pass