from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
from optuna import Trial
import optuna
import torch
from torch import nn, Tensor

from mlmodels import BaseMLModel
from .pytorch_model import PytorchModel
from patientdata import PatientData

class CnnSimple(PytorchModel):
    class InternalModel(nn.Module):
        def __init__(
            self,
            output_chn1_1: int = 10,
            kernel1_1: int = 3,
            relu1_1: bool = False,
            pool1_1: int = 0,
            output_chn1_2: int = 10,
            kernel1_2: int = 3,
            relu1_2: bool = False,
            pool1_2: int = 0,
            output_chn2_1: int = 10,
            kernel2_1: int = 3,
            relu2_1: bool = False,
            pool2_1: int = 0,
            output_chn2_2: int = 10,
            kernel2_2: int = 3,
            relu2_2: bool = False,
            pool2_2: int = 0,
            linear_layer: int = 2
        ) -> None:
            super().__init__()
            dimension1 = 22 - (kernel1 - 1)
            diemnsion2 = 22 - (kernel2 - 1)
            linear_size = (dimension1 * dimension1 * output_chn1) + (diemnsion2 * diemnsion2 * output_chn2)
            
            inter_linear1 = 2000
            inter_linear2 = 500
            
            avg_layers = [
                nn.Conv2d(1, output_chn1, kernel1)
            ]
            
            std_layers = [
                nn.Conv2d(1, output_chn2, kernel2)
            ]
            
            joined_layers = [
                nn.Linear(linear_size, 500), 
                nn.Linear(500, 2),
                nn.Softmax(1)
            ]
            
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
            
        if self.use_gpu:
            dataset_x = (dataset_x[0].cuda(), dataset_x[1].cuda())
            dataset_y = dataset_y.cuda() if dataset_y is not None else None
                    
        return dataset_x, dataset_y
    
    
    def objective(self, trial: Trial, dataset: List[PatientData]) -> BaseMLModel:
        epoch = trial.suggest_categorical("epoch", [100, 150, 200, 250, 300])
        output_chn1 = trial.suggest_int("output_chn1", 2, 15)
        kernel1 = trial.suggest_int("kernel1", 2, 5)
        output_chn2 = trial.suggest_int("output_chn2", 2, 15)
        kernel2 = trial.suggest_int("kernel2", 2, 5)
        
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.TrialPruned('Duplicate parameter set')
    
        model_copy = CnnSimple()
        model_copy.initialize_model(
            **{
                "epoch": epoch,
                "output_chn1": output_chn1,
                "kernel1": kernel1,
                "output_chn2": output_chn2,
                "kernel2": kernel2,
            }
        )
        
        model_copy.k_fold(dataset)
        
        return model_copy
        
        
if __name__ == "__main__":
    pass