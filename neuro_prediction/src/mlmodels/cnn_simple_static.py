from __future__ import annotations
from typing import Any, List, Tuple

from optuna import Trial
import torch
from torch import nn, Tensor
import numpy as np

from .pytorch_model import PytorchModel


class CnnSimpleStatic(PytorchModel):
    class InternalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stack = nn.Sequential(
                nn.Conv2d(1, 10, 3),
                nn.Flatten(),
                nn.Linear(4000, 2),
                nn.Softmax(1)
            )
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_simple_static", self.InternalModel)
        self.model = None
        self.param = None
        
        
    def extract_data(self, dataset_x: List[Any], dataset_y: List[Any]) -> Tuple[Tuple[Tensor, ...], Tensor]:
        dataset_x = torch.stack(list(map(lambda x: x[2], dataset_x))).to(torch.float32)
        if dataset_y is not None:
            dataset_y = torch.stack(list(map(lambda x: x[0], dataset_y))).to(torch.float32)
            
        if self.use_gpu:
            dataset_x = dataset_x.cuda()
            dataset_y = dataset_y.cuda() if dataset_y is not None else None
        
        return (dataset_x,), dataset_y
    
    
    def objective(self, trial: Trial) -> float:
        return super().objective(trial)
    
    
    

if __name__ == "__main__":
    pass