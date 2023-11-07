from __future__ import annotations
from typing import List, Tuple, Dict
from numpy.typing import NDArray

import dill
import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as metrics
from .base_mlmodel import BaseMLModel
from patientdata.data_enum import PatientOutcome, PatientSex

class SVMModel(BaseMLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
        
    def vectorize_fc(self, fc: NDArray) -> NDArray:
        output = []
        
        for i in range(len(fc) - 1):
            for j in range(i + 1, len(fc[i])):
                output.append(fc[i][j])
        
        return output
    
    
    def train_model(self, dataset: List[Tuple[NDArray, NDArray, Dict[int | bool | float | PatientOutcome | PatientSex]]]) -> None:
        x_label = [None] * len(dataset)
        y_label = [None] * len(dataset)
        data_split = None
        
        for i in range(len(dataset)):
            x_label[i] = np.concatenate([self.vectorize_fc(dataset[i][0]), self.vectorize_fc(dataset[i][1])])
            y_label[i] = int(dataset[i][2]["outcome"])
        
        data_split = self.get_data_split(x_label, y_label)
        
        output_acc = []
        
        for split in data_split:
            models = []
            models_acc = []
            test_x = [None] * len(split["test"])
            test_y = [None] * len(split["test"])
            
            for i in range(len(split["test"])):
                test_x[i] = x_label[split["test"][i]]
                test_y[i] = y_label[split["test"][i]]
            
            for train_val in split["train_val"]:
                train_x = [None] * len(train_val["train"])
                train_y = [None] * len(train_val["train"])
                val_x = [None] * len(train_val["val"])
                val_y = [None] * len(train_val["val"])
                
                for i in range(len(train_val["train"])):
                    train_x[i] = x_label[train_val["train"][i]]
                    train_y[i] = y_label[train_val["train"][i]]
                
                for i in range(len(train_val["val"])):
                    val_x[i] = x_label[train_val["val"][i]]
                    val_y[i] = y_label[train_val["val"][i]]
                    
                curr_model = SVC(**(self.model.get_params()))
                
                curr_model.fit(train_x, train_y)
                
                val_preds = curr_model.predict(val_x)
                val_acc = metrics.accuracy_score(val_y, val_preds)
                
                models.append(curr_model)
                models_acc.append(val_acc)
                
            best_model = models[np.argmax(models_acc)]
            test_preds = best_model.predict(test_x)
            
            output_acc.append(metrics.accuracy_score(test_y, test_preds))
            
        return np.mean(output_acc) # TODO: not sure what to return here
    
    
    def predict_result(self, avg_fc: NDArray, std_fc: NDArray, meta: Dict[str, int | bool | float | PatientOutcome | PatientSex]) -> PatientOutcome:
        pass
    
    
    def save_model(self, filename: str) -> None:
        with open(filename, "wb") as dill_file:
            dill.dump(self.model, dill_file)
    
    
    def load_model(self, filename: str) -> None:
        with open(filename, "rb") as dill_file:
            self.model = dill.load(dill_file)
    
    
    def initialize_model(self, **kwargs) -> None:
        self.model = SVC(**kwargs)


if __name__ == "__main__":
    pass