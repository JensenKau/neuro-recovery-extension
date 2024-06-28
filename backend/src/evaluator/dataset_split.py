from __future__ import annotations
from typing import List, Any


class DatasetSplit:
    def __init__(self, test: List[Any], train: List[Any] | DatasetSplit) -> None:
        self.test = None
        self.train = None
        
        
    def get_test(self) -> List[Any]:
        return self.test
    
    
    def get_train(self) -> List[Any] | DatasetSplit:
        return self.train


if __name__ == "__main__":
    pass