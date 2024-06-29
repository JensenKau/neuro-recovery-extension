from __future__ import annotations
from typing import List, Any

from src.patientdata.patient_data import PatientData

class DatasetSplit:
    def __init__(self, test: List[PatientData], train: List[PatientData] | DatasetSplit) -> None:
        self.test = None
        self.train = None
        
        
    def get_test(self) -> List[PatientData]:
        return self.test
    
    
    def get_train(self) -> List[PatientData] | DatasetSplit:
        return self.train


if __name__ == "__main__":
    pass