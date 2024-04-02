from __future__ import annotations
from typing import Any, List, Tuple
import math

from optuna import Trial
import optuna
import torch
from torch import nn, Tensor
import numpy as np

from src.patientdata.patient_data import PatientData
from src.mlmodels.pytorch_models.pytorch_model import PytorchModel
from src.mlmodels.base_mlmodel import BaseMLModel


class CnnSimpleStatic2_3(PytorchModel):
    class InternalModel(nn.Module):
        def __init__(
            self, 
            out_chn1: int = 5,
            kernel1: int = 3,
            stride1: int = 1,
            relu1: bool = False,
            dropout1: bool = False,
            out_chn2: int = 5,
            kernel2: int = 3,
            stride2: int = 1,
            relu2: bool = False,
            dropout2: bool = False,
            linear_layers: int = 2
        ) -> None:
            super().__init__()
            dimension0 = 22
            padding1 = math.ceil((21 * stride1 + kernel1 - dimension0) / 2)
            dimension1 = math.floor(((dimension0 + 2 * padding1 - (kernel1 - 1) - 1) / stride1) + 1)
            padding2 = math.ceil((21 * stride2 + kernel2 - dimension1) / 2)
            dimension2 = math.floor(((dimension1 + 2 * padding2 - (kernel2 - 1) - 1) / stride2) + 1)
            linear_size = dimension2 * dimension2 * out_chn2
            
            layers = [
                nn.Conv2d(1, out_chn1, kernel1, stride1, padding1),
                nn.ReLU() if relu1 else None,
                nn.Dropout2d(0.2) if dropout1 else None,
                nn.Conv2d(out_chn1, out_chn2, kernel2, stride2, padding2),
                nn.ReLU() if relu2 else None,
                nn.Dropout2d(0.2) if dropout2 else None,
                nn.Flatten()
            ] + {
                1: [nn.Linear(linear_size, 2)],
                2: [nn.Linear(linear_size, 1000), nn.Linear(1000, 2)],
                3: [nn.Linear(linear_size, 1000), nn.Linear(1000, 250), nn.Linear(250, 2)],
            }[linear_layers] + [
                nn.Softmax(1)
            ]
                        
            self.stack = nn.Sequential(*list(filter(lambda x: x is not None, layers)))
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_simple_static_2_3", self.InternalModel)
        
        
    def extract_data(self, dataset_x: List[Any], dataset_y: List[Any]) -> Tuple[Tuple[Tensor, ...], Tensor]:
        dataset_x = torch.stack(list(map(lambda x: x[2], dataset_x))).to(torch.float32)
        if dataset_y is not None:
            dataset_y = torch.stack(list(map(lambda x: x[0], dataset_y))).to(torch.float32)
            
        if self.use_gpu:
            dataset_x = dataset_x.cuda()
            dataset_y = dataset_y.cuda() if dataset_y is not None else None
        
        return (dataset_x,), dataset_y
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        out_chn1 = trial.suggest_int("out_chn1", 2, 10)
        kernel1 = trial.suggest_int("kernel1", 2, 5)
        stride1 = trial.suggest_int("stride1", 1, kernel1)
        relu1 = trial.suggest_categorical("relu1", [True, False])
        dropout1 = trial.suggest_categorical("dropout1", [True, False])
        out_chn2 = trial.suggest_int("out_chn2", out_chn1 + 1, 20)
        kernel2 = trial.suggest_int("kernel2", 2, 5)
        stride2 = trial.suggest_int("stride2", 1, kernel2)
        relu2 = trial.suggest_categorical("relu2", [True, False])
        dropout2 = trial.suggest_categorical("dropout2", [True, False])
        linear_layers = trial.suggest_int("linear_layers", 1, 3)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnSimpleStatic2_3()
        model_copy.initialize_model(**trial.params)
        
        model_copy.k_fold(dataset)
        
        return model_copy
    
    

if __name__ == "__main__":
    pass