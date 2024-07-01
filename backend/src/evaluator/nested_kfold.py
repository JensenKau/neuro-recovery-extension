from __future__ import annotations
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold

from src.mlmodels.base_mlmodel import BaseMLModel
from src.patientdata.patient_data import PatientData
from src.evaluator.model_evaluator import ModelEvaluator
from src.evaluator.dataset_split import DatasetSplit
from src.evaluator.model_performance import ModelPerformance


class NestedKFold(ModelEvaluator):
    RANDOM_STATE = 12345
    
    
    def __init__(self, model: BaseMLModel) -> None:
        super().__init__(model)
        self.performance = None
        
        
    def split_data(self, dataset: List[PatientData]) -> List[DatasetSplit]:
        output = []
        x, y = self.convert_data_to_index(dataset)
        skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.RANDOM_STATE)
        
        for inner, test in skf1.split(x, y):
            test = self.convert_index_to_data(test, dataset)
            inner = self.convert_index_to_data(inner, dataset)
            inner_x, inner_y = self.convert_data_to_index(inner)
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.RANDOM_STATE)
            current = []
            
            for train, val in skf2.split(inner_x, inner_y):
                train = self.convert_index_to_data(train, inner)
                val = self.convert_index_to_data(val, inner)
                current.append(DatasetSplit(val, train))
                
            output.append(DatasetSplit(test, current))
        
        return output
    
        
    def evaluate_performance(self, dataset: List[PatientData], testset: List[PatientData] = None) -> None:
        for split in self.split_data(dataset):
            test = split.get_test()
            inner = split.get_train()
            inner_models = []
            inner_performance = []
            
            for inner_split in inner:
                model_copy = self.get_model_copy()
                val = inner_split.get_test()
                train = inner_split.get_train()
                
                model_copy.train_model(train, val)
                y_true, y_pred = self.get_true_pred(model_copy, val)
                inner_models.append(model_copy)
                inner_performance.append(ModelPerformance.generate_performance(y_true, y_pred))
                
            
    
    
    def get_performance(self, args: str = None) -> Dict[str, Dict[str, float]]:
        if self.performance is None:
            raise ValueError("Must evaluate performance first before getting the perfomance result")
    
    
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        if self.performance is None:
            raise ValueError("Must evaluate performance first before saving the perfomance result")


if __name__ == "__main__":
    pass