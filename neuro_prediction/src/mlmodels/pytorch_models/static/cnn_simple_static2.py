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


class CnnSimpleStatic2(StaticModel):
    class InternalModel(nn.Module):
        def __init__(
            self, 
            output_chn1: int = 5, 
            kernel1: int = 3,
            output_chn2: int = 10,
            kernel2: int = 3,
            linear_size1: int = 1000
        ) -> None:
            super().__init__()
            dimension1 = 22 - (kernel1 - 1)
            dimension2 = dimension1 - (kernel2 - 1)
            linear_size_start = dimension2 * dimension2 * output_chn2
            
            self.stack = nn.Sequential(
                nn.Conv2d(1, output_chn1, kernel1),
                nn.Conv2d(output_chn1, output_chn2, kernel2),
                nn.Flatten(),
                nn.Linear(linear_size_start, linear_size1),
                nn.Linear(linear_size1, 2),
                nn.Softmax(1)
            )
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_simple_static_2", self.InternalModel)
        self.model = None
        self.param = None
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        output_chn1 = trial.suggest_int("output_chn1", 2, 5)
        kernel1 = trial.suggest_int("kernel1", 2, 5)
        output_chn2 = trial.suggest_int("output_chn2", 6, 10)
        kernel2 = trial.suggest_int("kernel2", 2, 5)
        linear_size1 = trial.suggest_categorical("linear_size1", [100, 250, 500])
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnSimpleStatic2()
        model_copy.initialize_model(
            **{
                "epoch": epoch,
                "output_chn1": output_chn1,
                "kernel1": kernel1,
                "output_chn2": output_chn2,
                "kernel2": kernel2,
                "linear_size1": linear_size1
            }
        )
        
        model_copy.k_fold(dataset)
        
        return model_copy
    
    

if __name__ == "__main__":
    pass