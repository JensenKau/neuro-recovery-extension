from __future__ import annotations
from typing import List, Any, Tuple, Callable
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
import numpy as np

from base_mlmodel import BaseMLModel
from patientdata import PatientData


class PytorchModel(BaseMLModel):
    def __init__(self, model_name: str, model_class: Callable) -> None:
        super().__init__(f"pytorch_{model_name}")
        self.model_class = model_class
        self.model = None
        self.param = None
        
        
    @abstractmethod
    def extract_data(self, dataset_x: List[Any], dataset_y: List[Any]) -> Tuple[Tuple[Tensor, ...], Tensor]:
        pass
        
        
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        dataset_x, dataset_y = self.extract_data(dataset_x, dataset_y)
        
        self.model.train()
        for epoch in range(100):
            y_pred = self.model(*dataset_x)
            loss = loss_fn(y_pred, dataset_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
    def predict_result_aux(self, dataset_x: List[Any]) -> List[Tuple[float, float]]:
        output = [None] * len(dataset_x)
        dataset_x, _ = self.extract_data(dataset_x, None)
        
        self.model.eval()
        with torch.inference_mode():
            res = self.model(dataset_x)
            for i in range(len(res)):
                current_res = np.argmax(res[i]).tolist()
                output[i] = (current_res, res[i][current_res])
        
        return output
    
    
    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)
    
    
    def load_model(self, filename: str) -> None:
        self.model = self.model_class()
        self.model.load_state_dict(torch.load(filename))
    
    
    def initialize_model(self, **kwargs) -> None:
        self.model = self.model_class()
        
        
    def create_model_copy(self) -> BaseMLModel:
        model_copy = self.__class__()
        model_copy.initialize_model()
        return model_copy
    
        
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        dataset_x = [None] * len(dataset)
        dataset_y = [None] * len(dataset)
        
        for i in range(len(dataset)):
            dataset_x[i] = dataset[i].get_fcs()
            meta = dataset[i].get_numberised_meta_data()
            outcome = meta["outcome"]
            cpc = meta["cpc"]
            if outcome != 0:
                y_outcome = [0.0, 0.0]
                y_outcome[int(outcome - 1)] = 1.0
                y_outcome = torch.tensor(y_outcome)

                y_cpc = [0.0, 0.0, 0.0, 0.0, 0.0]
                y_cpc[int(cpc - 1)] = 1.0
                y_cpc = torch.tensor(y_cpc)
                
                dataset_y[i] = (y_outcome, y_cpc)
                    
        return dataset_x, dataset_y
    
    
    def dataset_y_classification_num(self, dataset_y: List[Any]) -> List[int]:
        return np.argmax(list(map(lambda x: x.numpy(), dataset_y)), 1).tolist()
    
    
    def get_save_file_extension(self) -> str:
        return "pt"
        

if __name__ == "__main__":
    pass