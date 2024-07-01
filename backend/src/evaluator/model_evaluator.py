from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from src.patientdata.patient_data import PatientData
from src.mlmodels.base_mlmodel import BaseMLModel

class ModelEvaluator(ABC):
    def __init__(self, model: BaseMLModel) -> None:
        super().__init__()
        self.model = model
        
    
    def convert_data_to_index(self, dataset: List[PatientData]) -> Tuple[List[int], List[int]]:
        output_x = [i for i in range(len(dataset))]
        output_y = [int(data.get_numberised_meta_data()["outcome"]) for data in dataset]
        return output_x, output_y
    
    
    def convert_index_to_data(self, index: List[int], dataset: List[PatientData]) -> List[PatientData]:
        output = []
        for i in index:
            output.append(dataset[i])
        return output
    
    
    def get_model_copy(self) -> BaseMLModel:
        return self.model.create_model_copy()
    
    
    def get_true_pred(self, model: BaseMLModel, dataset: List[PatientData]) -> Tuple[List[int], List[int]]:
        y_true = [int(data.get_numberised_meta_data()["outcome"]) for data in dataset]
        y_pred = [int(res[0]) for res in model.predict_result(dataset)]
        return y_true, y_pred
                
    
    @abstractmethod
    def evaluate_performance(self, dataset: List[PatientData], testset: List[PatientData] = None) -> None:
        pass
    
    
    @abstractmethod
    def get_performance(self, args: str = None) -> Dict[str, Dict[str, float]]:
        pass
    
    
    @abstractmethod
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        pass
    
    
    @abstractmethod
    def save_model(self, folder: str) -> None:
        pass
    

if __name__ == "__main__":
    pass