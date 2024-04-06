from __future__ import annotations
from typing import List, Any, Tuple
import math

import numpy as np
from optuna import Trial
import optuna
import torch
from torch import nn, Tensor

from src.mlmodels.base_mlmodel import BaseMLModel
from src.mlmodels.pytorch_models.hybrid.hybrid_model import HybridModel
from src.patientdata.patient_data import PatientData

class CnnHybrid(HybridModel):
    class InternalModel(nn.Module):
        def __init__(
            self,
            output_chn1_1: int = 5,
            kernel1_1: int = 3,
            relu1_1: bool = False,
            pool1_1: int = 0,
            output_chn2_1: int = 5,
            kernel2_1: int = 3,
            relu2_1: bool = False,
            pool2_1: int = 0,
            output_chn3_1: int = 5,
            kernel3_1: int = 5,
            relu3_1: int = False,
            pool3_1: int = 5,
            linear_layer: int = 2
        ) -> None:
            super().__init__()

            dimension1_1 = 22 - (kernel1_1 - 1)
            dimension1_2 = math.floor(((dimension1_1 - (pool1_1 - 1) - 1) / pool1_1) + 1)
            
            dimension2_1 = 22 - (kernel2_1 - 1)
            dimension2_2 = math.floor(((dimension2_1 - (pool2_1 - 1) - 1) / pool2_1) + 1)
            
            dimension3_1 = 22 - (kernel3_1 - 1)
            dimension3_2 = math.floor(((dimension3_1 - (pool3_1 - 1) - 1) / pool3_1) + 1)
            
            linear_size = (dimension1_2 ** 2 * output_chn1_1) + (dimension2_2 ** 2 * output_chn2_1) + (dimension3_2 ** 2 * output_chn3_1)
            
            avg_layers = [
                nn.Conv2d(1, output_chn1_1, kernel1_1),
                nn.ReLU() if relu1_1 else None,
                nn.MaxPool2d(pool1_1) if pool1_1 > 1 else None
            ]
            
            std_layers = [
                nn.Conv2d(1, output_chn2_1, kernel2_1),
                nn.ReLU() if relu2_1 else None,
                nn.MaxPool2d(pool2_1) if pool2_1 > 1 else None
            ]
            
            static_layers = [
                nn.Conv2d(1, output_chn3_1, kernel3_1),
                nn.ReLU() if relu3_1 else None,
                nn.MaxPool2d(pool3_1) if pool3_1 > 1 else None
            ]
            
            joined_layers = {
                1: [nn.Linear(linear_size, 2)],
                2: [nn.Linear(linear_size, 500), nn.Linear(500, 2)],
                3: [nn.Linear(linear_size, 2000), nn.Linear(2000, 500), nn.Linear(500, 2)]
            }[linear_layer]
            joined_layers.append(nn.Softmax(1))
            
            self.avg_stack = nn.Sequential(*list(filter(lambda x: x is not None, avg_layers)))
            self.std_stack = nn.Sequential(*list(filter(lambda x: x is not None, std_layers)))
            self.static_stack = nn.Sequential(*list(filter(lambda x: x is not None, static_layers)))
            self.joined_stack = nn.Sequential(*list(filter(lambda x: x is not None, joined_layers)))
            
            
        def forward(self, avg: Tensor, std: Tensor, static: Tensor) -> Tensor:
            avg_out = self.avg_stack(torch.unsqueeze(avg, 1))
            std_out = self.std_stack(torch.unsqueeze(std, 1))
            static_out = self.static_stack(torch.unsqueeze(static, 1))
            
            joined_out = torch.cat((torch.flatten(avg_out, 1), torch.flatten(std_out, 1), torch.flatten(static_out, 1)), 1)
            joined_out = self.joined_stack(joined_out)
            
            return joined_out
        
    
    def __init__(self) -> None:
        super().__init__("cnn_hybrid", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        
        output_chn1_1 = trial.suggest_int("output_chn1_1", 2, 20)
        kernel1_1 = trial.suggest_int("kernel1_1", 2, 5)
        relu1_1 = trial.suggest_categorical("relu1_1", [True, False])
        pool1_1 = trial.suggest_int("pool1_1", 1, 5)
        
        output_chn2_1 = trial.suggest_int("output_chn2_1", 2, 20)
        kernel2_1 = trial.suggest_int("kernel2_1", 2, 5)
        relu2_1 = trial.suggest_categorical("relu2_1", [True, False])
        pool2_1 = trial.suggest_int("pool2_1", 1, 5)
        
        output_chn3_1 = trial.suggest_int("output_chn3_1", 2, 20)
        kernel3_1 = trial.suggest_int("kernel3_1", 2, 5)
        relu3_1 = trial.suggest_categorical("relu3_1", [True, False])
        pool3_1 = trial.suggest_int("pool3_1", 1, 5)
        
        linear_layer = trial.suggest_int("linear_layer", 1, 3)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnHybrid()
        model_copy.initialize_model(**trial.params)
        
        model_copy.k_fold(dataset)
        
        return model_copy
        
        
if __name__ == "__main__":
    pass