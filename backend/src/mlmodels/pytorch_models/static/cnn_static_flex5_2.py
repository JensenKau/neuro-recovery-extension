from __future__ import annotations
from typing import Any, List, Dict
import math

from optuna import Trial
import torch
from torch import nn, Tensor

from src.mlmodels.pytorch_models.static.static_model import StaticModel


class CnnStaticFlex5_2(StaticModel):
    class InternalModel(nn.Module):
        def __init__(
            self, 
            cnn_conf: List[Dict[str, str | int | float | None]] = None,
            linear_conf: List[int] = None
        ) -> None:
            super().__init__()
            layers = []
            
            if cnn_conf is None:
                cnn_conf = [{
                    "out_chn": 5,
                    "kernel": 3,
                    "stride": 1,
                    "activation": None,
                    "dropout": 0.0
                }] 
            if linear_conf is None:
                linear_conf = []
                
            previous_dimension = 22
            for i in range(len(cnn_conf)):
                padding = math.ceil((21 * cnn_conf[i]["stride"] + cnn_conf[i]["kernel"] - previous_dimension) / 2)
                previous_dimension = math.floor(((previous_dimension + 2 * padding - (cnn_conf[i]["kernel"] - 1) - 1) / cnn_conf[i]["stride"]) + 1)
                
                layers.append(nn.Conv2d(cnn_conf[i - 1]["out_chn"] if i > 0 else 1, cnn_conf[i]["out_chn"], cnn_conf[i]["kernel"], cnn_conf[i]["stride"], padding))
                layers.append(nn.ReLU() if cnn_conf[i]["activation"] == "relu" else (nn.LeakyReLU() if cnn_conf[i]["activation"] == "leaky" else None))
                layers.append(nn.BatchNorm2d(cnn_conf[i]["out_chn"]))
                layers.append(nn.Dropout2d(cnn_conf[i]["dropout"]) if cnn_conf[i]["dropout"] > 0.0 else None)
                
            layers.append(nn.Flatten())
            
            previous_size = previous_dimension * previous_dimension * cnn_conf[len(cnn_conf) - 1]["out_chn"]
            for i in range(len(linear_conf)):
                layers.append(nn.Linear(previous_size, linear_conf[i]))
                previous_size = linear_conf[i]
            layers.append(nn.Linear(previous_size, 2))
            
            layers.append(nn.Softmax(1))
                        
            self.stack = nn.Sequential(*list(filter(lambda x: x is not None, layers)))
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_static_flex_5_2", self.InternalModel)
    
    
    def objective(self, trial: Trial) -> Dict[str, Any]:
        epoch = trial.suggest_categorical("epoch", [100, 200, 300, 400, 500])
        cnn_layers = trial.suggest_int("cnn_layers", 1, 5)
        linear_layers = trial.suggest_int("linear_layers", 1, 3)
        
        cnn_conf = [None] * cnn_layers
        linear_conf = [1000, 100][:linear_layers - 1]
        
        for i in range(len(cnn_conf)):
            out_chn_lower = cnn_conf[i - 1]["out_chn"] + 1 if i > 0 else 2
            out_chn_upper = (i + 1) * 10
            
            out_chn = trial.suggest_int(f"out_chn{i}", out_chn_lower, out_chn_upper)
            kernel = trial.suggest_int(f"kernel{i}", 2, 5)
            stride = trial.suggest_int(f"stride{i}", 1, kernel)
            activation = trial.suggest_categorical(f"activation{i}", ["relu", "leaky", None])
            dropout = trial.suggest_categorical(f"dropout{i}", [0.0, 0.1, 0.2])
            
            cnn_conf[i] = {
                "out_chn": out_chn,
                "kernel": kernel,
                "stride": stride,
                "activation": activation,
                "dropout": dropout
            }
        
        return {
            "epoch": epoch,
            "cnn_conf": cnn_conf,
            "linear_conf": linear_conf
        }
    
    

if __name__ == "__main__":
    pass