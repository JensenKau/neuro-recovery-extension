from __future__ import annotations
from typing import List, Any, Dict

import sklearn.metrics as metrics
import numpy as np

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
    
    
    def get_performance(self) -> Dict[str, float]:
        return {
            "Acc": self.acc,
            "Pre": self.pre,
            "Rec": self.rec,
            "F1": self.f1,
            "ROC": self.roc
        }
    
    
    def __str__(self) -> str:
        return f"Acc: {self.acc}\t Pre: {self.pre}\t Rec: {self.rec}\t F1: {self.f1}\t ROC: {self.roc}"
    
    
    @classmethod
    def generate_performance(cls, y_true: List[int], y_pred: List[int]) -> ModelPerformance:
        return ModelPerformance(
            acc=metrics.accuracy_score(y_true, y_pred),
            pre=metrics.precision_score(y_true, y_pred, average="binary", zero_division=0.0),
            rec=metrics.recall_score(y_true, y_pred, average="binary", zero_division=0.0),
            f1=metrics.f1_score(y_true, y_pred, average="binary", zero_division=0.0),
            roc=metrics.roc_auc_score(y_true, y_pred)
        )
        
        
    @classmethod
    def avg_performance(cls, performances: List[ModelPerformance]) -> ModelPerformance:
        return ModelPerformance(
            acc=float(np.mean(list(map(lambda x: x.get_acc(), performances)), axis=0)),
            pre=float(np.mean(list(map(lambda x: x.get_pre(), performances)), axis=0)),
            rec=float(np.mean(list(map(lambda x: x.get_rec(), performances)), axis=0)),
            f1=float(np.mean(list(map(lambda x: x.get_f1(), performances)), axis=0)),
            roc=float(np.mean(list(map(lambda x: x.get_roc(), performances)), axis=0))
        )
        
        
    @classmethod
    def std_performance(cls, performances: List[ModelPerformance]) -> ModelPerformance:
        return ModelPerformance(
            acc=float(np.std(list(map(lambda x: x.get_acc(), performances)), axis=0)),
            pre=float(np.std(list(map(lambda x: x.get_pre(), performances)), axis=0)),
            rec=float(np.std(list(map(lambda x: x.get_rec(), performances)), axis=0)),
            f1=float(np.std(list(map(lambda x: x.get_f1(), performances)), axis=0)),
            roc=float(np.std(list(map(lambda x: x.get_roc(), performances)), axis=0))
        )

if __name__ == "__main__":
    pass