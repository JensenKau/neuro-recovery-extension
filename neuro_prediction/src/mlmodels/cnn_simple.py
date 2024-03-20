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
        super().__init__()
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
        
        
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        avg, std = self.dataset_x_tensor(dataset_x)
        dataset_y = torch.stack(dataset_y)
        
        self.model.train()
        
        for epoch in range(100):
            y_pred = self.model(avg.to(torch.float32), std.to(torch.float32))
            loss = loss_fn(y_pred, dataset_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
   
    def predict_result_aux(self, dataset_x: List[Any]) -> List[Tuple[float, float]]:
        output = [None] * len(dataset_x)
        avg, std = self.dataset_x_tensor(dataset_x)
                
        self.model.eval()
        
        with torch.inference_mode():
            res = self.model(avg.to(torch.float32), std.to(torch.float32))
            for i in range(len(res)):
                current_res = np.argmax(res[i]).tolist()
                output[i] = (current_res, res[i][current_res])
                
        return output
    
   
    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)
    
   
    def load_model(self, filename: str) -> None:
        self.model = self.InternalModel()
        self.model.load_state_dict(torch.load(filename))
    
   
    def initialize_model(self, **kwargs) -> None:
        self.model = self.InternalModel()
            
   
    def create_model_copy(self) -> BaseMLModel:
        model_copy = CnnSimple()
        model_copy.initialize_model()
        return model_copy
    
   
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        dataset_x = [None] * len(dataset)
        dataset_y = [None] * len(dataset)
        
        for i in range(len(dataset)):
            avg_fc, std_fc, static_fc = dataset[i].get_fcs()
            res = dataset[i].get_numberised_meta_data()["outcome"]
            dataset_x[i] = (torch.from_numpy(avg_fc), torch.from_numpy(std_fc))
            if res is not None:
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