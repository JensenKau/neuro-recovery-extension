from __future__ import annotations
from typing import Dict, Tuple, Callable, Any, List

from numpy.typing import NDArray
import scipy.io
import numpy as np
import mne
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class PatientEEGData:
    LOW_PASS = 0.1
    HIGH_PASS = 30.0
    BRAIN_REGION = [
        "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", 
        "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", 
        "Oz", "F9"
    ]
    
    def __init__(self, eeg_data: Dict[str, NDArray], num_points: int, sampling_frequency: int, utility_frequency: int, start_time: int, end_time: int, preprocess: bool = True) -> None:
        self.eeg_data = eeg_data
        self.num_points = num_points
        self.sampling_frequency = sampling_frequency
        self.utility_frequency = utility_frequency
        self.start_time = start_time
        self.end_time = end_time
        
        if preprocess:
            self.apply_filter()
            self.apply_resampling(128)
            self.apply_normalization()
        
        
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
        return func(line) if func(line) is not None else (
            args if len(args) > 1 else args[0]
        )
    
    
    @classmethod
    def load_eeg_data(cls, header_file: str, content_file: str, preprocess: bool = True) -> PatientEEGData:
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
            end_time=end_time,
            preprocess=preprocess
        )
        
    
    @classmethod
    def load_eeg_datas(cls, header_files: List[str], content_files: List[str]) -> PatientEEGData:
        eegs = [None] * len(header_files)
        
        for i in range(len(header_files)):
            current_eeg = cls.load_eeg_data(header_files[i], content_files[i], False)
            eegs[i] = current_eeg
        
        return cls.merge_eeg_data(eegs)
        
    
    @classmethod
    def merge_eeg_data(cls, eegs: List[PatientEEGData], resampling_frequency: int = 128, preprocess: bool = True) -> PatientEEGData:
        eegs = sorted(eegs, key=lambda x: x.get_start_time())
        time = [None] * len(eegs)
        wave = [[None] * len(eegs) for _ in range(len(cls.BRAIN_REGION))]
        output_wave = {}
        start_time = eegs[0].get_start_time()
        end_time = eegs[-1].get_end_time()
        num_points = (end_time - start_time + 1) * resampling_frequency
        
        for i in range(len(eegs)):
            eeg = eegs[i]
            eeg.apply_resampling(resampling_frequency)
            regions = eeg.get_eeg_data()
            time[i] = np.linspace(eeg.get_start_time(), eeg.get_end_time(), eeg.get_num_points(), dtype=np.float64)
            
            for j in range(len(cls.BRAIN_REGION)):
                region = cls.BRAIN_REGION[j]
                if region in regions:
                    wave[j][i] = regions[region]
                    
        for i in range(len(cls.BRAIN_REGION)):
            current_time = []
            current_wave = []
            
            for j in range(len(time)):
                if wave[i][j] is not None:
                    current_time.append(time[j])
                    current_wave.append(wave[i][j])

            if len(current_time) > 0:
                current_time = np.concatenate(current_time, axis=None, dtype=np.float64)
                current_wave = np.concatenate(current_wave, axis=None, dtype=np.float64)
                interpolator = interp1d(current_time, current_wave)
                output_wave[cls.BRAIN_REGION[i]] = interpolator(np.linspace(start_time, end_time, num_points))
                
        return PatientEEGData(
            output_wave,
            num_points,
            resampling_frequency,
            eegs[0].get_utility_frequency(),
            start_time,
            end_time,
            preprocess
        )

    
    
    def convert_eeg_to_table(self) -> NDArray:
        output = [None] * len(self.BRAIN_REGION)
        
        for i in range(len(self.BRAIN_REGION)):
            region = self.BRAIN_REGION[i]
            if region in self.eeg_data:
                output[i] = self.eeg_data[region]
            else:
                output[i] = np.zeros(self.num_points)
        
        return np.asarray(output, dtype=np.float64)
    
    
    def convert_table_to_eeg(self, table: NDArray) -> Dict[str, NDArray]:
        output = {}
        
        for i in range(len(self.BRAIN_REGION)):
            region = self.BRAIN_REGION[i]
            if region in self.eeg_data:
                output[region] = table[i]
        
        return output
    
    
    def apply_filter(self) -> None:
        table = self.convert_eeg_to_table()
                
        if self.LOW_PASS <= self.utility_frequency <= self.HIGH_PASS:
            table = mne.filter.notch_filter(table, self.sampling_frequency, self.utility_frequency, n_jobs=4, verbose="error")
        table = mne.filter.filter_data(table, self.sampling_frequency, self.LOW_PASS, self.HIGH_PASS, n_jobs=4, verbose="error")
        
        self.eeg_data = self.convert_table_to_eeg(table)
    
    
    def apply_normalization(self) -> None:
        table = self.convert_eeg_to_table()
        
        min_value = np.min(table)
        max_value = np.max(table)
        if min_value != max_value:
            table = 2.0 / (max_value - min_value) * (table - 0.5 * (min_value + max_value))
        else:
            table = 0 * table
        
        self.eeg_data = self.convert_table_to_eeg(table)
    
    
    def apply_resampling(self, resampling_frequency: int) -> None:
        table = self.convert_eeg_to_table()
        
        lcm = np.lcm(int(round(self.sampling_frequency)), int(round(resampling_frequency)))
        up = int(round(lcm / self.sampling_frequency))
        down = int(round(lcm / resampling_frequency))
        resampling_frequency = self.sampling_frequency * up / down
        table = scipy.signal.resample_poly(table, up, down, axis=1)
        
        self.eeg_data = self.convert_table_to_eeg(table)
        self.sampling_frequency = int(resampling_frequency)
        self.num_points = (self.end_time - self.start_time + 1) * self.sampling_frequency
       
       
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
    