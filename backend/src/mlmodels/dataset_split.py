from __future__ import annotations
from typing import List, Tuple, Any

class DatasetSplit:
    def __init__(self) -> None:
        self.test_set = None
        self.validation_sets = []
        self.train_sets = []
        
    def get_test_set(self) -> Tuple[List[Any], List[Any]]:
        return self.test_set
    
    def set_test_set(self, test_set: Tuple[List[Any], List[Any]]) -> None:
        self.test_set = test_set
    
    def get_validation_sets(self) -> List[Tuple[List[Any], List[Any]]]:
        return self.validation_sets
    
    def add_validation_set(self, new_set: Tuple[List[Any], List[Any]]) -> None:
        self.validation_sets.append(new_set)
    
    def get_train_sets(self) -> List[Tuple[List[Any], List[Any]]]:
        return self.train_sets
    
    def add_train_set(self, new_set: Tuple[List[Any], List[Any]]) -> None:
        self.train_sets.append(new_set)

if __name__ == "__main__":
    pass