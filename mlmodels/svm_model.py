from __future__ import annotations
from typing import List, Tuple
from numpy.typing import NDArray

from .base_mlmodel import BaseMLModel
from patientdata.data_enum import PatientOutcome, PatientSex

class SVMModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
    
    def train_model(self, dataset: List[Tuple[NDArray, NDArray, List[int | bool | float | PatientOutcome | PatientSex]]]) -> None:
        pass
    
    def predict_result(self, data: Tuple[NDArray, NDArray, List[int | bool | float | PatientOutcome | PatientSex]]) -> None:
        pass
    
    def save_model(self, filename: str) -> None:
        pass
    
    def load_model(self, filename: str) -> None:
        pass
    
    def initialize_model(self, **kwargs) -> None:
        pass

if __name__ == "__main__":
    pass