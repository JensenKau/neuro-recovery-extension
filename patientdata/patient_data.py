from __future__ import annotations
from typing import Dict, List, Tuple
from numpy.typing import NDArray

import math

from .eeg_data import PatientEEGData
from .meta_data import PatientMetaData
from .connectivity_data import PatientConnectivityData
from .data_enum import PatientOutcome, PatientSex

class PatientData:
    def __init__(self, eeg: PatientEEGData, meta: PatientMetaData) -> None:
        # self.eeg = eeg
        self.meta = meta
        self.connectivity = PatientConnectivityData.load_patient_connectivity(eeg)
        
    @classmethod
    def load_patient_data(cls, meta_file: str, header_file: str, eeg_file: str) -> PatientData:
        return PatientData(
            eeg=PatientEEGData.load_eeg_data(header_file, eeg_file),
            meta=PatientMetaData.load_patient_meta_data(meta_file)
        )
        
    def get_patient_id(self) -> int:
        return self.meta.get_patient_id()
    
    def get_hospital(self) -> str:
        return self.meta.get_hospital()
    
    def get_age(self) -> int:
        return self.meta.get_age()
    
    def get_sex(self) -> PatientSex:
        return self.meta.get_sex()
    
    def get_rosc(self) -> float:
        return self.meta.get_rosc()
    
    def get_ohca(self) -> bool:
        return self.meta.get_ohca()
    
    def get_shockable_rhythm(self) -> bool:
        return self.meta.get_shockable_rhythm()
    
    def get_ttm(self) -> int:
        return self.meta.get_ttm()
    
    def get_outcome(self) -> PatientOutcome:
        return self.meta.get_outcome()
    
    def get_cpc(self) -> int:
        return self.meta.get_cpc()
    
    def get_eeg_data(self) -> Dict[str, NDArray]:
        return self.eeg.get_eeg_data()
    
    def get_num_points(self) -> int:
        return self.eeg.get_num_points()
    
    def get_sampling_frequency(self) -> int:
        return self.eeg.get_sampling_frequency()
    
    def get_utility_frequency(self) -> int:
        return self.eeg.get_utility_frequency()
    
    def get_start_time(self) -> int:
        return self.eeg.get_start_time()
    
    def get_end_time(self) -> int:
        return self.eeg.get_end_time()
    
    def get_avg_fc(self) -> NDArray:
        return self.connectivity.get_avg_fc()
    
    def get_std_fc(self) -> NDArray:
        return self.connectivity.get_std_fc()
    
    def get_meta_data(self) -> Dict[str, int | bool | float | PatientOutcome | PatientSex]:
        return {
            "patient": self.get_patient_id(),
            "age": self.get_age(),
            "sex": self.get_sex(),
            "rosc": self.get_rosc(),
            "ohca": self.get_ohca(),
            "shockable rhythm": self.get_shockable_rhythm(),
            "ttm": self.get_ttm(),
            "outcome": self.get_outcome(),
            "cpc": self.get_cpc(),
            "start time": self.get_start_time(),
            "end time": self.get_end_time()
        }
    
    def get_numberised_data(self) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
        meta = [
            self.get_age(),
            0 if self.get_sex() == PatientSex.MALE else 1,
            self.get_rosc(),
            1 if self.get_ohca() else 0,
            1 if self.get_shockable_rhythm() else 0,
            self.get_ttm(),
            self.get_start_time(),
            self.get_end_time()
        ]
        
        res = [
            1 if self.get_outcome() == PatientOutcome.GOOD else 0,
            self.get_cpc()
        ]
        
        for i in range(len(meta)):
            if math.isnan(meta[i]):
                meta[i] = 0
            meta[i] = float(meta[i])
            
        for i in range(len(res)):
            if math.isnan(res[i]):
                res[i] = 0
            res[i] = float(res[i])
            
        return self.get_avg_fc(), self.get_std_fc(), meta, res

if __name__ == "__main__":
    pass