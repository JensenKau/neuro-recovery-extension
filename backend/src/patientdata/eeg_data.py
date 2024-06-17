from __future__ import annotations
from typing import Dict, Tuple, Callable, Any
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
    def read_time(cls, time: str) -> int:
        hour, minute, second = list(map(lambda x: int(x), time.split(":")))
        return (hour * 3600) + (minute * 60) + second
    
    
    @classmethod
    def parse_sampling_rate(cls, item: str) -> Tuple[int, int]:
        split = item.split()
        return (int(split[2]), int(split[3])) if len(split) == 4 else None
    
    
    @classmethod
    def parse_utitlity_rate(cls, item: str) -> int:
        return int(item.split(": ")[1]) if item.lower().startswith("#utility frequency") else None
    
    
    @classmethod
    def parse_start_time(cls, item: str) -> int:
        return cls.read_time(item.split(": ")[1]) if item.lower().startswith("#start time") else None
    
    
    @classmethod
    def parse_end_time(cls, item: str) -> int:
        return cls.read_time(item.split(": ")[1]) if item.lower().startswith("#end time") else None
    
    
    @classmethod
    def parse_eeg_info(cls, item: str) -> Tuple[float, int, str]:
        split = item.split()
        return (float(split[2].split("/")[0]), int(split[4]), split[8]) if len(split) > 4 else None
    
    
    @classmethod
    def parse_item(cls, func: Callable[[str], Any], line: str, *args) -> Any:
        return func(line) if func(line) is not None else args
    
    
    @classmethod
    def load_eeg_data(cls, header_file: str, content_file: str) -> PatientEEGData:
        raw_eeg = scipy.io.loadmat(content_file)["val"]
        eeg_data = {}
        sampling_frequncy, num_points = 0, 0
        utility_frequency = 0
        start_time, end_time = 0, 0
        
        with open(header_file, "r", encoding="utf-8") as file:
            file = file.read().strip().split("\n")
            
            for i in range(len(file)):
                line = file[i]
                
                sampling_frequncy, num_points = cls.parse_item(cls.parse_sampling_rate, line, sampling_frequncy, num_points)
                utility_frequency = cls.parse_item(cls.parse_utitlity_rate, line, utility_frequency)
                start_time = cls.parse_item(cls.parse_start_time, line, start_time)
                end_time = cls.parse_item(cls.parse_end_time, line, end_time)
                gain, offset, channel = cls.parse_item(cls.parse_eeg_info, line, None, None, None)
                
                if gain is not None and offset is not None and channel is not None:
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
    