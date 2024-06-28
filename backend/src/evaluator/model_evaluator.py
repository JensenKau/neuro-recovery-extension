from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from src.patientdata.patient_data import PatientData


class ModelEvaluator(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    
    def convert_data_to_index(self, dataset: List[PatientData]) -> Tuple[List[int], List[int]]:
        output_x = [i for i in range(len(dataset))]
        output_y = [int(data.get_numberised_meta_data()["outcome"]) for data in dataset]
        return output_x, output_y
    
    
    def convert_index_to_data(self, index: List[int], dataset: List[PatientData]) -> List[PatientData]:
        output = []
        
        for i in index:
            output.append(dataset[i])
        
        return output
                
    
    @abstractmethod
    def evaluate_performance(self, dataset: List[PatientData], testset: List[PatientData] = None) -> None:
        pass
    
    
    @abstractmethod
    def get_performance(self, **kwargs) -> Dict[str, float]:
        pass
    
    
    @abstractmethod
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        pass
    

if __name__ == "__main__":
    pass