from __future__ import annotations
from typing import Dict, List
import csv

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
        self.validation = None
        self.test = None


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
            
        self.validation = val_performances
        self.test = test_performances
    
    
    def get_performance(self, args: str = None) -> Dict[str, Dict[str, float]]:
        if self.validation is None or self.test is None:
            raise ValueError("Must evaluate performance first before getting the perfomance result")
        
        output = {}
        
        if args is None:
            for i in range(len(self.validation)):
                output[f"fold_{i + 1}_val"] = self.validation[i].get_performance()
                output[f"fold_{i + 1}_test"] = self.test[i].get_performance()
        
        if args is None or args == "avg":
            output["avg_val"] = ModelPerformance.avg_performance(self.validation).get_performance()
            output["avg_test"] = ModelPerformance.avg_performance(self.test).get_performance()
        
        if args is None or args == "std":
            output["std_val"] = ModelPerformance.std_performance(self.validation).get_performance()
            output["std_test"] = ModelPerformance.std_performance(self.test).get_performance()
            
        return output
    
    
    def save_performance(self, file: str, add_extension: bool = True) -> None:
        if self.validation is None or self.test is None:
            raise ValueError("Must evaluate performance first before saving the perfomance result")
        
        if add_extension and not file.endswith(".csv"):
            file = f"{file}.csv"
            
        fields = list(self.validation[0].get_performance().keys())
    

if __name__ == "__main__":
    pass