from __future__ import annotations
from typing import Any, Dict

from optuna import Trial
import torch
from torch import nn, Tensor

from src.mlmodels.pytorch_models.static.static_model import StaticModel


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
    
    
    def objective(self, trial: Trial) -> Dict[str, Any]:
        output_chn = trial.suggest_int("output_chn", 2, 10)
        kernel = trial.suggest_int("kernel", 1, 10)
        
        return trial.params
    
    
    

if __name__ == "__main__":
    pass