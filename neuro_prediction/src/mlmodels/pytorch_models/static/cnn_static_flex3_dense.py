from __future__ import annotations
from typing import Any, List, Tuple, Dict
import math

from optuna import Trial
import optuna
import torch
from torch import nn, Tensor
import numpy as np

from src.patientdata.patient_data import PatientData
from src.mlmodels.pytorch_models.static.static_model import StaticModel
from src.mlmodels.base_mlmodel import BaseMLModel


class CnnStaticFlex3_Dense(StaticModel):
    class InternalModel(nn.Module):
        def __init__(
            self, 
            cnn_conf: List[Dict[str, str | int | float | bool | None]] = None,
            linear_conf: List[int] = None
        ) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            
            current_channel = 1
            for i in range(len(cnn_conf)):
                padding = math.ceil((21 * cnn_conf[i]["stride"] + cnn_conf[i]["kernel"] - 22) / 2)
                dimension = math.floor(((22 + 2 * padding - (cnn_conf[i]["kernel"] - 1) - 1) / cnn_conf[i]["stride"]) + 1)
                avg_kernel = dimension - 21
                self.layers.append(nn.Sequential(*list(filter(lambda x: x is not None, [
                    nn.Conv2d(current_channel, cnn_conf[i]["out_chn"], cnn_conf[i]["kernel"], cnn_conf[i]["stride"], padding),
                    nn.ReLU() if cnn_conf[i]["activation"] == "relu" else (nn.LeakyReLU() if cnn_conf[i]["activation"] == "leaky" else None),
                    nn.BatchNorm2d(cnn_conf[i]["out_chn"]),
                    nn.AvgPool2d(avg_kernel, 1) if avg_kernel > 1 else None,
                    nn.Dropout2d() if cnn_conf[i]["dropout"] else None,
                ]))))
                current_channel += cnn_conf[i]["out_chn"]
            
            linear = [nn.Flatten()]
            previous_size = 22 * 22 * current_channel
            for i in range(len(linear_conf)):
                linear.append(nn.Linear(previous_size, linear_conf[i]))
                previous_size = linear_conf[i]
            linear.append(nn.Linear(previous_size, 2))
            linear.append(nn.Softmax(1))
            self.layers.append(nn.Sequential(*linear))
                        
        
        def forward(self, static_fc: Tensor) -> Tensor:
            previous = torch.unsqueeze(static_fc, 1)
            for i in range(len(self.layers)):
                if i < len(self.layers) - 1:
                    res = self.layers[i](previous)
                    previous = torch.cat((previous, res), 1)
                else:
                    return self.layers[i](previous)
    
    
    
    
    def __init__(self) -> None:
        super().__init__("cnn_static_flex_3_dense", self.InternalModel)
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300, 350, 400, 450, 500])
        cnn_layers = trial.suggest_int("cnn_layers", 1, 3)
        linear_layers = trial.suggest_int("linear_layers", 1, 5)
        
        cnn_conf = [None] * cnn_layers
        linear_conf = [2000, 1000, 500, 250][:linear_layers - 1]
        
        for i in range(len(cnn_conf)):
            out_chn = trial.suggest_int(f"out_chn{i}", 1, (i + 1) * (i + 1) * 10)
            kernel = trial.suggest_int(f"kernel{i}", 1, 7)
            stride = trial.suggest_int(f"stride{i}", 1, kernel)
            activation = trial.suggest_categorical(f"activation{i}", [None, "relu", "leaky"])
            dropout = trial.suggest_categorical(f"dropout{i}", [True, False])
            
            cnn_conf[i] = {
                "out_chn": out_chn,
                "kernel": kernel,
                "stride": stride,
                "activation": activation,
                "dropout": dropout
            }
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnStaticFlex3_Dense()
        model_copy.initialize_model(**{
            "epoch": epoch,
            "cnn_conf": cnn_conf,
            "linear_conf": linear_conf
        })
        
        model_copy.k_fold(dataset)
        
        return model_copy
    
    

if __name__ == "__main__":
    pass