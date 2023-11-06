from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
from numpy.typing import NDArray

from sklearn.model_selection import StratifiedKFold
from patientdata.data_enum import PatientOutcome, PatientSex

class BaseMLModel(ABC):
    SEED1 = 123
    SEED2 = 456
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train_model(self, dataset: List[Tuple[NDArray, NDArray, Dict[int | bool | float | PatientOutcome | PatientSex]]]) -> None:
        pass
    
    @abstractmethod
    def predict_result(self, avg_fc: NDArray, std_fc: NDArray, meta: Dict[str, int | bool | float | PatientOutcome | PatientSex]) -> PatientOutcome:
        pass
    
    @abstractmethod
    def save_model(self, filename: str) -> None:
        pass
    
    @abstractmethod
    def load_model(self, filename: str) -> None:
        pass
    
    @abstractmethod
    def initialize_model(self, **kwargs) -> None:
        pass

    def get_data_split(self, dataset_x: List[Any], dataset_y: List[Any]) -> List[Dict[str, List[int] | Dict[str, List[int]]]]:
        skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=BaseMLModel.SEED1)
        output = []
        
        for train_val, test in skf1.split(dataset_x, dataset_y):
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=BaseMLModel.SEED2)
            train_val_list = []
            dataset_x_train_val = [None] * len(train_val)
            dataset_y_train_val = [None] * len(train_val)
            
            for i in range(len(train_val)):
                dataset_x_train_val[i] = dataset_x[i]
                dataset_y_train_val[i] = dataset_y[i]
                
            for train, val in skf2.split(dataset_x_train_val, dataset_y_train_val):
                train_val_list.append({"train": train, "val": val})
                
            output.append({"train_val": train_val_list, "test": test})
            
        return output
            

if __name__ == "__main__":
    pass