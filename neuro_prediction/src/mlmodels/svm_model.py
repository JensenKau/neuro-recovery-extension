from __future__ import annotations
from typing import Any, List, Tuple
from numpy.typing import NDArray

import dill
import numpy as np
from sklearn.svm import SVC

from patientdata.patient_data import PatientData
from .base_mlmodel import BaseMLModel

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
    
    
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        self.model.fit(dataset_x, dataset_y)
    
    
    def predict_result_aux(self, dataset_x: List[Any]) -> List[float]:
        return self.model.predict(dataset_x)
    
    
    def create_model_copy(self) -> BaseMLModel:
        model = SVMModel()
        model.initialize_model(**self.model.get_params())
        return model
    
    
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        dataset_x = [None] * len(dataset)
        dataset_y = [None] * len(dataset)
        
        for i in range(len(dataset)):
            avg_fc, std_fc, meta, res = dataset[i].get_numberised_data()
            avg_fc = self.vectorize_fc(avg_fc)
            std_fc = self.vectorize_fc(std_fc)
            dataset_x[i] = avg_fc + std_fc
            dataset_y[i] = res[0]
        
        return dataset_x, dataset_y
    
    
    def save_model(self, filename: str) -> None:
        with open(filename, "wb") as dill_file:
            dill.dump(self.model, dill_file)
    
    
    def load_model(self, filename: str) -> None:
        with open(filename, "rb") as dill_file:
            self.model = dill.load(dill_file)
    
    
    def initialize_model(self, **kwargs) -> None:
        self.model = SVC(**kwargs)


    def dataset_y_classification_num(self, dataset_y: List[Any]) -> List[int]:
        return dataset_y


if __name__ == "__main__":
    pass