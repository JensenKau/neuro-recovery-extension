from __future__ import annotations
from typing import List, Any

import sklearn.metrics as metrics

class ModelPerformance:
    def __init__(self, acc: float, pre: float, rec: float, f1: float, roc: float) -> None:
        self.acc = acc
        self.pre = pre
        self.rec = rec
        self.f1 = f1
        self.roc = roc
        
    def get_acc(self) -> float:
        return self.acc
    
    def get_pre(self) -> float:
        return self.pre
    
    def get_rec(self) -> float:
        return self.rec
    
    def get_f1(self) -> float:
        return self.f1
    
    def get_roc(self) -> float:
        return self.roc
    
    @classmethod
    def generate_performance(cls, y_true: List[Any], y_pred: List[Any]) -> ModelPerformance:
        return ModelPerformance(
            acc=metrics.accuracy_score(y_true, y_pred),
            pre=metrics.precision_score(y_true, y_pred, average="binary"),
            rec=metrics.recall_score(y_true, y_pred, average="binary"),
            f1=metrics.f1_score(y_true, y_pred, average="binary"),
            roc=metrics.roc_auc_score(y_true, y_pred)
        )

if __name__ == "__main__":
    pass