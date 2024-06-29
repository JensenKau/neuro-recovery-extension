from __future__ import annotations
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold

from src.patientdata.patient_data import PatientData
from src.evaluator.model_evaluator import ModelEvaluator
from src.evaluator.dataset_split import DatasetSplit
from src.evaluator.model_performance import ModelPerformance
from src.mlmodels.base_mlmodel import BaseMLModel


class KFold(ModelEvaluator):
    RANDOM_STATE = 12345
    
    
    def __init__(self, model: BaseMLModel) -> None:
        super().__init__(model)
        self.performance = None


    def split_data(self, dataset: List[PatientData]) -> List[DatasetSplit]:
        x, y = self.convert_data_to_index(dataset)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.RANDOM_STATE)
        output = []
        
        for train, val in skf.split(x, y):
            output.append(DatasetSplit(
                self.convert_index_to_data(val, dataset),
                self.convert_index_to_data(train, dataset)
            ))
            
        return output
        
        
    def evaluate_performance(self, dataset: List[PatientData], testset: List[PatientData] = None) -> None:
        if testset is None:
            raise ValueError("testset parameter must not be None")
        
        val_performances = []
        test_performances = []
        
        for split in self.split_data(dataset):
            model_copy = self.get_model_copy()
            train = split.get_train()
            val = split.get_test()
            
            model_copy.train_model(train, val)
            
            y_true, y_pred = self.get_true_pred(model_copy, val)
            val_performances.append(ModelPerformance.generate_performance(y_true, y_pred))
            
            y_true, y_pred = self.get_true_pred(model_copy, testset)
            test_performances.append(ModelPerformance.generate_performance(y_true, y_pred))
            
        self.performance = (val_performances, test_performances)
    
    
    def get_performance(self, **kwargs) -> Dict[str, float]:
        pass
    
    
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        pass
    

if __name__ == "__main__":
    pass