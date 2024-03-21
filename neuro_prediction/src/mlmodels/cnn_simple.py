from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
from optuna import Trial
import torch
from torch import nn, Tensor

from mlmodels import BaseMLModel
from .pytorch_model import PytorchModel
from patientdata import PatientData

class CnnSimple(PytorchModel):
    class InternalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.avg_stack = nn.Sequential(
                nn.Conv2d(1, 10, 3)
            )
            
            self.std_stack = nn.Sequential(
                nn.Conv2d(1, 10, 3)
            )
            
            self.joined_stack = nn.Sequential(
                nn.Linear(8000, 500), 
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
        self.model = None
        self.params = None
        
    
    def dataset_x_tensor(self, dataset_x: List[Any]) -> Tuple[Tensor, Tensor]:
        avg = [None] * len(dataset_x)
        std = [None] * len(dataset_x)
        
        for i in range(len(dataset_x)):
            avg[i], std[i] = dataset_x[i]
            
        avg = torch.stack(avg)
        std = torch.stack(std)
        
        return avg, std
        
    def extract_data(self, dataset_x: List[Any], dataset_y: List[Any]) -> Tuple[Tuple[Tensor] | Tensor]:
        dataset_x = (
            torch.stack(list(map(lambda x: x[0], dataset_x))).to(torch.float32),
            torch.stack(list(map(lambda x: x[1], dataset_x))).to(torch.float32)
        )
        if dataset_y is not None:
            dataset_y = torch.stack(list(map(lambda x: x[0], dataset_y))).to(torch.float32)
        
        return dataset_x, dataset_y
    
    
    def objective(self, trial: Trial) -> float:
        return None
        
        
if __name__ == "__main__":
    pass