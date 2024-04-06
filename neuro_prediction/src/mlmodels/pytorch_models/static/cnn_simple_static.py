from __future__ import annotations
from typing import Any, List, Tuple

from optuna import Trial
import optuna
import torch
from torch import nn, Tensor
import numpy as np

from src.patientdata.patient_data import PatientData
from src.mlmodels.pytorch_models.static.static_model import StaticModel
from src.mlmodels.base_mlmodel import BaseMLModel


class CnnSimpleStatic(StaticModel):
    class InternalModel(nn.Module):
        def __init__(
            self, 
            output_chn: int = 10, 
            kernel: int = 3,
        ) -> None:
            super().__init__()
            dimension = 22 - (kernel - 1)
            linear_size = dimension * dimension * output_chn
            
            self.stack = nn.Sequential(
                nn.Conv2d(1, output_chn, kernel),
                nn.Flatten(),
                nn.Linear(linear_size, 2),
                nn.Softmax(1)
            )
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_simple_static", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        output_chn = trial.suggest_int("output_chn", 2, 10)
        kernel = trial.suggest_int("kernel", 2, 10)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
        
        model_copy = CnnSimpleStatic()
        model_copy.initialize_model(**{"epoch": epoch, "output_chn": output_chn, "kernel": kernel})
                
        model_copy.k_fold(dataset)
        
        return model_copy
    
    
    

if __name__ == "__main__":
    pass