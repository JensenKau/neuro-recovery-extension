from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
from optuna import Trial
import optuna
import torch
from torch import nn, Tensor

from src.mlmodels import BaseMLModel
from src.mlmodels.pytorch_models.dynamic.dynamic_model import DynamicModel
from src.patientdata.patient_data import PatientData

class CnnSimple(DynamicModel):
    class InternalModel(nn.Module):
        def __init__(
            self,
            output_chn1: int = 10,
            kernel1: int = 3,
            output_chn2: int = 10,
            kernel2: int = 3
        ) -> None:
            super().__init__()
            dimension1 = 22 - (kernel1 - 1)
            diemnsion2 = 22 - (kernel2 - 1)
            linear_size = (dimension1 * dimension1 * output_chn1) + (diemnsion2 * diemnsion2 * output_chn2)
            
            self.avg_stack = nn.Sequential(
                nn.Conv2d(1, output_chn1, kernel1)
            )
            
            self.std_stack = nn.Sequential(
                nn.Conv2d(1, output_chn2, kernel2)
            )
            
            self.joined_stack = nn.Sequential(
                nn.Linear(linear_size, 500), 
                nn.Linear(500, 2),
                nn.Softmax(1)
            )
            
        def forward(self, avg: Tensor, std: Tensor) -> Tensor:
            avg_out = self.avg_stack(torch.unsqueeze(avg, 1))
            std_out = self.std_stack(torch.unsqueeze(std, 1))
            
            joined_out = torch.cat((torch.flatten(avg_out, 1), torch.flatten(std_out, 1)), 1)
            joined_out = self.joined_stack(joined_out)
            
            return joined_out
        
    
    def __init__(self) -> None:
        super().__init__("cnn_simple", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        output_chn1 = trial.suggest_int("output_chn1", 2, 15)
        kernel1 = trial.suggest_int("kernel1", 2, 5)
        output_chn2 = trial.suggest_int("output_chn2", 2, 15)
        kernel2 = trial.suggest_int("kernel2", 2, 5)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnSimple()
        model_copy.initialize_model(
            **{
                "epoch": epoch,
                "output_chn1": output_chn1,
                "kernel1": kernel1,
                "output_chn2": output_chn2,
                "kernel2": kernel2,
            }
        )
        
        model_copy.k_fold(dataset)
        
        return model_copy
        
        
if __name__ == "__main__":
    pass