from __future__ import annotations
from typing import List, Tuple, Dict
from numpy.typing import NDArray

import dill
import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as metrics
from .base_mlmodel import BaseMLModel
from patientdata.data_enum import PatientOutcome, PatientSex

class SVMModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
        
    def vectorize_fc(self, fc: NDArray) -> NDArray:
        output = []
        
        for i in range(len(fc) - 1):
            for j in range(i + 1, len(fc[i])):
                output.append(fc[i][j])
        
        return output
    
    
    
    
    
    def save_model(self, filename: str) -> None:
        with open(filename, "wb") as dill_file:
            dill.dump(self.model, dill_file)
    
    
    def load_model(self, filename: str) -> None:
        with open(filename, "rb") as dill_file:
            self.model = dill.load(dill_file)
    
    
    def initialize_model(self, **kwargs) -> None:
        self.model = SVC(**kwargs)


if __name__ == "__main__":
    pass