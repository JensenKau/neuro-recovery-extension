from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from mlmodels import BaseMLModel
from patientdata import PatientData

class CnnSimple(BaseMLModel):
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
                nn.Linear(500, 300), # NOTE: 500 is just placeholder number
                nn.Linear(300, 100),
                nn.Linear(100, 50),
                nn.Linear(50, 2),
                nn.Softmax()
            )
            
        def forward(self, x: Tensor) -> Tensor:
            avg_out = self.avg_stack(x[0])
            std_out = self.std_stack(x[1])
            
            joined_out = torch.cat((torch.flatten(avg_out), torch.flatten(std_out)))
            joined_out = self.joined_stack(joined_out)
            
            return joined_out
        
    
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.params = None
        
        
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        self.model.train()
        
        for epoch in range(100):
            y_pred = self.model(dataset_x)
            loss = loss_fn(y_pred, dataset_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
   
    def predict_result_aux(self, dataset_x: List[Any]) -> List[float]:
        output = [None] * len(dataset_x)
        
        self.model.eval()
        
        with torch.inference_mode():
            res = self.model(torch.tensor(dataset_x))
            
            for i in range(len(res)):
                output[i] = np.argmax(res[i])
            
        return output
    
   
    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)
    
   
    def load_model(self, filename: str) -> None:
        self.model = CnnSimple.InternalModel()
        self.model.load_state_dict(torch.load(filename))
    
   
    def initialize_model(self, **kwargs) -> None:
        self.model = CnnSimple.InternalModel()
    
   
    def create_model_copy(self) -> BaseMLModel:
        model_copy = CnnSimple()
        model_copy.initialize_model()
        return model_copy
    
   
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        dataset_x = [None] * len(dataset)
        dataset_y = [None] * len(dataset)
        
        for i in range(len(dataset)):
            avg_fc, std_fc, meta, res = dataset[i].get_numberised_data()
            dataset_x[i] = torch.from_numpy(np.array([avg_fc, std_fc]))
            dataset_y[i] = [0.0, 0.0]
            dataset_y[i][res[0]] = 1.0
            dataset_y[i] = torch.tensor(dataset_y[i])
        
        return dataset_x, dataset_y

if __name__ == "__main__":
    pass