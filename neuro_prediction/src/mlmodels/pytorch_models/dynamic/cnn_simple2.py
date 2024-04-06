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

class CnnSimple2(DynamicModel):
    class InternalModel(nn.Module):
        def __init__(
            self,
            output_chn1_1: int = 5,
            kernel1_1: int = 3,
            relu1_1: bool = False,
            output_chn1_2: int = 5,
            kernel1_2: int = 3,
            relu1_2: bool = False,
            pool1_1: int = 0,
            output_chn2_1: int = 5,
            kernel2_1: int = 3,
            relu2_1: bool = False,
            output_chn2_2: int = 5,
            kernel2_2: int = 3,
            relu2_2: bool = False,
            pool2_1: int = 0,
            linear_layer: int = 2
        ) -> None:
            super().__init__()

            dimension1_1 = 22 - (kernel1_1 - 1)
            dimension1_2 = dimension1_1 - (kernel1_2 - 1)
            dimension1_3 = dimension1_2 if pool1_1 == 0 else int(((dimension1_2 - (pool1_1 - 1) - 1) / pool1_1) + 1)
            
            dimension2_1 = 22 - (kernel2_1 - 1)
            dimension2_2 = dimension2_1 - (kernel2_2 - 1)
            dimension2_3 = dimension2_2 if pool2_1 == 0 else int(((dimension2_2 - (pool2_1 - 1) - 1) / pool2_1) + 1)
            
            linear_size = (dimension1_3 * dimension1_3 * output_chn1_2) + (dimension2_3 * dimension2_3 * output_chn2_2)
            
            avg_layers = [
                nn.Conv2d(1, output_chn1_1, kernel1_1),
                nn.ReLU() if relu1_1 else None,
                nn.Conv2d(output_chn1_1, output_chn1_2, kernel1_2),
                nn.ReLU() if relu1_2 else None,
                nn.MaxPool2d(pool1_1) if pool1_1 > 0 else None
            ]
            
            std_layers = [
                nn.Conv2d(1, output_chn2_1, kernel2_1),
                nn.ReLU() if relu2_1 else None,
                nn.Conv2d(output_chn2_1, output_chn2_2, kernel2_2),
                nn.ReLU() if relu2_2 else None,
                nn.MaxPool2d(pool2_1) if pool2_1 > 0 else None
            ]
            
            joined_layers = {
                1: [nn.Linear(linear_size, 2)],
                2: [nn.Linear(linear_size, 500), nn.Linear(500, 2)],
                3: [nn.Linear(linear_size, 2000), nn.Linear(2000, 500), nn.Linear(500, 2)]
            }[linear_layer]
            joined_layers.append(nn.Softmax(1))
            
            self.avg_stack = nn.Sequential(*list(filter(lambda x: x is not None, avg_layers)))
            self.std_stack = nn.Sequential(*list(filter(lambda x: x is not None, std_layers)))
            self.joined_stack = nn.Sequential(*list(filter(lambda x: x is not None, joined_layers)))
            
            
        def forward(self, avg: Tensor, std: Tensor) -> Tensor:
            avg_out = self.avg_stack(torch.unsqueeze(avg, 1))
            std_out = self.std_stack(torch.unsqueeze(std, 1))
            
            joined_out = torch.cat((torch.flatten(avg_out, 1), torch.flatten(std_out, 1)), 1)
            joined_out = self.joined_stack(joined_out)
            
            return joined_out
        
    
    def __init__(self) -> None:
        super().__init__("cnn_simple_2", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        
        output_chn1_1 = trial.suggest_int("output_chn1_1", 2, 10)
        kernel1_1 = trial.suggest_int("kernel1_1", 2, 5)
        relu1_1 = trial.suggest_categorical("relu1_1", [True, False])
        
        output_chn1_2 = trial.suggest_int("output_chn1_2", 2, 15)
        kernel1_2 = trial.suggest_int("kernel1_2", 2, 5)
        relu1_2 = trial.suggest_categorical("relu1_2", [True, False])
        pool1_1 = trial.suggest_int("pool1_1", 0, 5)
        
        output_chn2_1 = trial.suggest_int("output_chn2_1", 2, 10)
        kernel2_1 = trial.suggest_int("kernel2_1", 2, 5)
        relu2_1 = trial.suggest_categorical("relu2_1", [True, False])
        
        output_chn2_2 = trial.suggest_int("output_chn2_2", 2, 15)
        kernel2_2 = trial.suggest_int("kernel2_2", 2, 5)
        relu2_2 = trial.suggest_categorical("relu2_2", [True, False])
        pool2_1 = trial.suggest_int("pool2_1", 0, 5)
        
        linear_layer = trial.suggest_int("linear_layer", 1, 3)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnSimple2()
        model_copy.initialize_model(**trial.params)
        
        model_copy.k_fold(dataset)
        
        return model_copy
        
        
if __name__ == "__main__":
    pass