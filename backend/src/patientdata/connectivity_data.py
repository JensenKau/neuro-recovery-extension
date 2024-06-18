from __future__ import annotations
from typing import Tuple
from numpy.typing import NDArray

from src.patientdata.eeg_data import PatientEEGData

import numpy as np
import scipy
import mne
from nilearn.connectome import ConnectivityMeasure
import warnings
import math

class PatientConnectivityData:
    BRAIN_REGION = [
        "Fp1", "Fp2", "F7", "F8", "F3", 
        "F4", "T3", "T4", "C3", "C4", 
        "T5", "T6", "P3", "P4", "O1", 
        "O2", "Fz", "Cz", "Pz", "Fpz", 
        "Oz", "F9"
    ]
    
    def __init__(self, avg_fc: NDArray, std_fc: NDArray, static_fc: NDArray) -> None:
        self.avg_fc = avg_fc
        self.std_fc = std_fc
        self.static_fc = static_fc
    
    
    @classmethod
    def load_patient_connectivity(cls, eeg_data: PatientEEGData) -> PatientConnectivityData:
        actual_eeg = eeg_data.get_eeg_data()
        num_points = eeg_data.get_num_points()
        sampling_frequency = eeg_data.get_sampling_frequency()
        utility_frequency = eeg_data.get_utility_frequency()
        regions = PatientConnectivityData.BRAIN_REGION
        organized_data = [None] * len(regions)

        for i in range(len(regions)):
            if regions[i] in actual_eeg:
                organized_data[i] = actual_eeg[regions[i]]
            else:
                organized_data[i] = np.zeros(num_points)
                
        organized_data, sampling_frequency = PatientConnectivityData.preprocess_data(np.array(organized_data), sampling_frequency, utility_frequency)
        avg_fc, std_fc, static_fc = PatientConnectivityData.gnerate_fc(organized_data, sampling_frequency)
        
        return PatientConnectivityData(avg_fc, std_fc, static_fc)
    
    
    @classmethod
    def gnerate_fc(cls, eeg_data: NDArray, sampling_frequency: int) -> Tuple[NDArray, NDArray, NDArray]:
        sample_size = 60
        window_size = sample_size * sampling_frequency
        shift_size = int(window_size * 0.1)
        index, pointer = 0, 0
        avg_fc, std_fc, static_fc = None, None, None
        measure = ConnectivityMeasure(kind="correlation")
        temp_fcs = [np.zeros((len(eeg_data), len(eeg_data)), dtype=np.float64)] * int(math.ceil(len(eeg_data[0]) / shift_size))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while index < len(eeg_data[0]):
                start = index
                end = int(min(start + window_size, len(eeg_data[0])))
                current_fc = measure.fit_transform(np.array([eeg_data[:, start:end]]).swapaxes(1, 2))[0]
                if not np.isnan(current_fc).any():
                    temp_fcs[pointer] = current_fc
                index += shift_size
                pointer += 1           
            avg_fc = np.mean(temp_fcs, axis=0)
            std_fc = np.std(temp_fcs, axis=0)
            static_fc = measure.fit_transform(np.array([eeg_data]).swapaxes(1, 2))[0]
        
        return avg_fc, std_fc, static_fc
    
    
    def get_avg_fc(self) -> NDArray:
        return self.avg_fc
    
    
    def get_std_fc(self) -> NDArray:
        return self.std_fc

    
    def get_static_fc(self) -> NDArray:
        return self.static_fc
    
    
    
    
if __name__ == "__main__":
    pass