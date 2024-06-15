from __future__ import annotations
from typing import Callable, Any, Tuple, List

import torch
from torch import Tensor

from src.mlmodels.pytorch_models.pytorch_model import PytorchModel


class DynamicModel(PytorchModel):
    def __init__(self, model_name: str, model_class: Callable[..., Any]) -> None:
        super().__init__(model_name, model_class)
        
        
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




if __name__ == "__main__":
    pass