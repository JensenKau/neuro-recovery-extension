from __future__ import annotations
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold

from src.mlmodels.base_mlmodel import BaseMLModel
from src.patientdata.patient_data import PatientData
from src.evaluator.model_evaluator import ModelEvaluator


class NestedKFold(ModelEvaluator):
    def __init__(self, model: BaseMLModel) -> None:
        super().__init__(model)
        
        
    def evaluate_performance(self, dataset: List[PatientData], testset: List[PatientData] = None) -> None:
        pass
    
    
    def get_performance(self, args: str = None) -> Dict[str, Dict[str, float]]:
        pass
    
    
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        pass


if __name__ == "__main__":
    pass