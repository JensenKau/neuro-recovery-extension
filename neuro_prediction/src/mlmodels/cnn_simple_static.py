from __future__ import annotations
from typing import Any, List, Tuple

from patientdata.patient_data import PatientData
from .base_mlmodel import BaseMLModel

import torch
from torch import nn, Tensor
import numpy as np


class CnnSimpleStatic(BaseMLModel):
    class InternalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stack = nn.Sequential(
                nn.Conv2d(1, 10, 3),
                nn.Flatten(),
                nn.Linear(4000, 2),
                nn.Softmax(0)
            )
            
        
        def forward(self, static_fc: Tensor) -> Tensor:
            return self.stack(torch.unsqueeze(static_fc, 1))
    
    
    
    
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.param = None
        
        
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        static_fc = torch.stack(dataset_x)
        dataset_y = torch.stack(dataset_y)
        
        self.model.train()
        for epoch in range(100):
            y_pred = self.model(static_fc.to(torch.float32))
            loss = loss_fn(y_pred, dataset_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    
    def predict_result_aux(self, dataset_x: List[Any]) -> List[float]:
        output = [None] * len(dataset_x)
        static_fc = torch.stack(dataset_x)
        
        self.model.eval()
        with torch.inference_mode():
            res = self.model(static_fc.to(torch.float32))
            for i in range(len(res)):
                output[i] = np.argmax(res[i]).tolist()
        
        return output
    
    
    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)
    
    
    def load_model(self, filename: str) -> None:
        self.model = self.InternalModel()
        self.model.load_state_dict(torch.load(filename))
    
    
    def initialize_model(self, **kwargs) -> None:
        self.model = self.InternalModel()
    
    
    def create_model_copy(self) -> BaseMLModel:
        model_copy = CnnSimpleStatic()
        model_copy.initialize_model()
        return model_copy
    
    
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        dataset_x = [None] * len(dataset)
        dataset_y = [None] * len(dataset)
        
        for i in range(len(dataset)):
            avg_fc, std_fc, static_fc = dataset[i].get_fcs()
            res = dataset[i].get_numberised_meta_data()["outcome"]
            dataset_x[i] = torch.from_numpy(static_fc)
            dataset_y[i] = [0.0, 0.0]
            dataset_y[i][int(res - 1)] = 1.0
            dataset_y[i] = torch.tensor(dataset_y[i])
                    
        return dataset_x, dataset_y
    
    
    def dataset_y_classification_num(self, dataset_y: List[Any]) -> List[int]:
        return np.argmax(list(map(lambda x: x.numpy(), dataset_y)), 1).tolist()
    
    
    def get_save_file_extension(self) -> str:
        return "pt"
    
    


if __name__ == "__main__":
    pass