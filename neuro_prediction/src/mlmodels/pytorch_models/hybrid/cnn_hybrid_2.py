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

class CnnHybrid1_2(HybridModel):
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

            linear1 = 22 - (kernel1_1 - 1)
            linear1 = math.floor(((linear1 - pool1_1) / pool1_1) + 1)
            linear1 = 22 * output_chn1_1 * linear1
            
            linear2 = 22 - (kernel2_1 - 1)
            linear2 = math.floor(((linear2 - pool2_1) / pool2_1) + 1)
            linear2 = 22 * output_chn2_1 * linear2
            
            linear3 = 22 - (kernel3_1 - 1)
            linear3 = math.floor(((linear3 - pool3_1) / pool3_1) + 1)
            linear3 = 22 * output_chn3_1 * linear3
            
            avg_layers = [
                nn.Conv2d(1, output_chn1_1, (1, kernel1_1)),
                nn.ReLU() if relu1_1 else None,
                nn.BatchNorm2d(output_chn1_1),
                nn.AvgPool2d((1, pool1_1)) if pool1_1 > 1 else None,
                nn.Flatten(),
                nn.Linear(linear1, 1500)
            ]
            
            std_layers = [
                nn.Conv2d(1, output_chn2_1, (1, kernel2_1)),
                nn.ReLU() if relu2_1 else None,
                nn.BatchNorm2d(output_chn2_1),
                nn.AvgPool2d((1, pool2_1)) if pool2_1 > 1 else None,
                nn.Flatten(),
                nn.Linear(linear2, 1500),
            ]
            
            static_layers = [
                nn.Conv2d(1, output_chn3_1, (1, kernel3_1)),
                nn.ReLU() if relu3_1 else None,
                nn.BatchNorm2d(output_chn3_1),
                nn.AvgPool2d((1, pool3_1)) if pool3_1 > 1 else None,
                nn.Flatten(),
                nn.Linear(linear3, 1500)
            ]
            
            joined_layers = {
                1: [nn.Linear(4500, 2)],
                2: [nn.Linear(4500, 500), nn.Linear(500, 2)],
                3: [nn.Linear(4500, 2000), nn.Linear(2000, 500), nn.Linear(500, 2)]
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
            
            joined_out = torch.cat((avg_out, std_out, static_out), 1)
            joined_out = self.joined_stack(joined_out)
            
            return joined_out
        
    
    def __init__(self) -> None:
        super().__init__("cnn_hybrid_1_2", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_int("epoch", 100, 500)
        
        output_chn1_1 = trial.suggest_int("output_chn1_1", 2, 20)
        kernel1_1 = trial.suggest_int("kernel1_1", 1, 5)
        relu1_1 = trial.suggest_categorical("relu1_1", [True, False])
        pool1_1 = trial.suggest_int("pool1_1", 1, 5)
        
        output_chn2_1 = trial.suggest_int("output_chn2_1", 2, 20)
        kernel2_1 = trial.suggest_int("kernel2_1", 1, 5)
        relu2_1 = trial.suggest_categorical("relu2_1", [True, False])
        pool2_1 = trial.suggest_int("pool2_1", 1, 5)
        
        output_chn3_1 = trial.suggest_int("output_chn3_1", 2, 20)
        kernel3_1 = trial.suggest_int("kernel3_1", 1, 5)
        relu3_1 = trial.suggest_categorical("relu3_1", [True, False])
        pool3_1 = trial.suggest_int("pool3_1", 1, 5)
        
        linear_layer = trial.suggest_int("linear_layer", 1, 3)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnHybrid1_2()
        model_copy.initialize_model(**trial.params)
        
        model_copy.k_fold(dataset)
        
        return model_copy
        
        
if __name__ == "__main__":
    pass