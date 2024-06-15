from __future__ import annotations
from typing import Any, Dict

from optuna import Trial
import torch
from torch import nn, Tensor

from src.mlmodels.pytorch_models.static.static_model import StaticModel


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
    
    
    def objective(self, trial: Trial) -> Dict[str, Any]:
        output_chn1 = trial.suggest_int("output_chn1", 2, 20)
        kernel1 = trial.suggest_int("kernel1", 1, 5)
        output_chn2 = trial.suggest_int("output_chn2", 2, 40)
        kernel2 = trial.suggest_int("kernel2", 1, 5)
        linear_size1 = trial.suggest_categorical("linear_size1", [100, 250, 500, 750, 1000])
        
        return trial.params
    
    

if __name__ == "__main__":
    pass