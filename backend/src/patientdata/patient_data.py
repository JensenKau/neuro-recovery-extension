from __future__ import annotations
from typing import Dict, List, Tuple
from numpy.typing import NDArray

import math
from enum import Enum

from src.patientdata.eeg_data import PatientEEGData
from src.patientdata.meta_data import PatientMetaData
from src.patientdata.connectivity_data import PatientConnectivityData
from src.patientdata.data_enum import PatientOutcome, PatientSex

class PatientData:
    def __init__(self, eeg: PatientEEGData, meta: PatientMetaData, keep_eeg: bool = False, keep_fc: bool = False) -> None:
        self.eeg = eeg
        self.meta = meta
        self.connectivity = PatientConnectivityData.load_patient_connectivity(eeg)
        
        if not keep_eeg:
            self.eeg.delete_eeg_data()
        
        if not keep_fc:
            self.connectivity.delete_dynamic_fc()


    @classmethod
    def load_patient_data(cls, meta_file: str, header_files: List[str], eeg_files: List[str], keep_eeg: bool = False, keep_fc: bool = False) -> PatientData:
        return PatientData(
            eeg=PatientEEGData.load_eeg_datas(header_files, eeg_files),
            meta=PatientMetaData.load_patient_meta_data(meta_file),
            keep_eeg=keep_eeg,
            keep_fc=keep_fc
        )
        
    
    @classmethod
    def load_patient_processed(cls, meta_file: str, merged_eeg_file: str, keep_eeg: bool = False, keep_fc: bool = False) -> PatientData:
        return PatientData(
            eeg=PatientEEGData.load_processed_eeg(merged_eeg_file),
            meta=PatientMetaData.load_patient_meta_data(meta_file),
            keep_eeg=keep_eeg,
            keep_fc=keep_fc
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
    
    
    def get_static_fc(self) -> NDArray:
        return self.connectivity.get_static_fc()

    
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
        

    def get_fcs(self) -> Tuple[NDArray, NDArray, NDArray]:
        return self.get_avg_fc(), self.get_std_fc(), self.get_static_fc()


    def get_numberised_meta_data(self) -> Dict[str, float]:
        output = self.get_meta_data()
        
        for key in output.keys():
            if isinstance(output[key], Enum):
                output[key] = output[key].value
            elif math.isnan(output[key]):
                output[key] = 0
            else:
                output[key] = float(output[key])
                
        return output




if __name__ == "__main__":
    pass