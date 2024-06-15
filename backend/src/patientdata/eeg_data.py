from __future__ import annotations
from typing import Dict, Tuple
from numpy.typing import NDArray

import scipy.io
import numpy as np

class PatientEEGData:    
    def __init__(self, eeg_data: Dict[str, NDArray], num_points: int, sampling_frequency: int, utility_frequency: int, start_time: int, end_time: int) -> None:
        self.eeg_data = eeg_data
        self.num_points = num_points
        self.sampling_frequency = sampling_frequency
        self.utility_frequency = utility_frequency
        self.start_time = start_time
        self.end_time = end_time
    
    
    @classmethod
    def load_eeg_data(cls, header_file: str, content_file: str) -> PatientEEGData:
        raw_eeg = scipy.io.loadmat(content_file)["val"]
        eeg_data = {}
        num_points = 0
        sampling_frequncy = 0
        utility_frequency = 0
        start_time = 0
        end_time = 0
        
        with open(header_file, "r", encoding="utf-8") as file:
            file = file.read().strip().split("\n")
            
            for i in range(len(file)):
                line = file[i].split()
                
                if i == 0:
                    sampling_frequncy = int(line[2])
                    num_points = int(line[3])
                    
                elif file[i][0] == "#":
                    line = file[i].split(": ")
                    if line[0].lower() == "#utility frequency":
                        utility_frequency = int(line[1])
                    elif line[0].lower() == "#start time":
                        time_list = line[1].split(":")
                        start_time = (int(time_list[0]) * 3600) + (int(time_list[1]) * 60) + int(time_list[2])
                    elif line[0].lower() == "#end time":
                        time_list = line[1].split(":")
                        end_time = (int(time_list[0]) * 3600) + (int(time_list[1]) * 60) + int(time_list[2])
                        
                else:
                    gain = float(line[2].split('/')[0])
                    offset = int(line[4])
                    channel = line[8]
                    eeg_data[channel] = (raw_eeg[i - 1].astype(np.float64) - offset) / gain
        
        return PatientEEGData(
            eeg_data=eeg_data,
            num_points=num_points,
            sampling_frequency=sampling_frequncy,
            utility_frequency=utility_frequency,
            start_time=start_time,
            end_time=end_time
        )
    
       
    def delete_eeg_data(self) -> None:
        self.eeg_data = None


    def get_eeg_data(self) -> Dict[str, NDArray]:
        return self.eeg_data

    
    def get_num_points(self) -> int:
        return self.num_points

    
    def get_sampling_frequency(self) -> int:
        return self.sampling_frequency

    
    def get_utility_frequency(self) -> int:
        return self.utility_frequency

    
    def get_start_time(self) -> int:
        return self.start_time

    
    def get_end_time(self) -> int:
        return self.end_time


    

if __name__ == "__main__":
    eeg_data = scipy.io.loadmat("D:\\neuro-recovery-prediction\\data\\training\\training\\0284\\0284_001_004_EEG.mat")["val"]
    