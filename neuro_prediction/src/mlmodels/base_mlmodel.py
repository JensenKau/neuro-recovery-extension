from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from enum import Enum
import os
import csv

import numpy as np
from sklearn.model_selection import StratifiedKFold

from patientdata.data_enum import PatientOutcome
from patientdata.patient_data import PatientData
from .model_performance import ModelPerformance
from .dataset_split import DatasetSplit

class BaseMLModel(ABC):
    SEED1 = 123
    SEED2 = 456
    NUM_SPLIT = 5
    
    class SAVE_MODE(Enum):
        BEST = 0
        ALL = 1
        NONE = 2
    
    
    def __init__(self) -> None:
        super().__init__()
        self.is_save_k_fold = self.SAVE_MODE.NONE
        self.save_folder = None


    def train_model(self, dataset: List[PatientData]) -> None:
        dataset_x, dataset_y = self.reshape_input(dataset)
        self.train_model_aux(dataset_x, dataset_y)

        
    def predict_result(self, dataset: List[PatientData]) -> List[Tuple[float, float]]:
        dataset_x, _ = self.reshape_input(dataset)
        return self.predict_result_aux(dataset_x)

    
    @abstractmethod
    def train_model_aux(self, dataset_x: List[Any], dataset_y: List[Any]) -> None:
        pass

    
    @abstractmethod
    def predict_result_aux(self, dataset_x: List[Any]) -> List[Tuple[float, float]]:
        pass

    
    @abstractmethod
    def save_model(self, filename: str) -> None:
        pass

    
    @abstractmethod
    def load_model(self, filename: str) -> None:
        pass

    
    @abstractmethod
    def initialize_model(self, **kwargs) -> None:
        pass

    
    @abstractmethod
    def create_model_copy(self) -> BaseMLModel:
        pass

    
    @abstractmethod
    def reshape_input(self, dataset: List[PatientData]) -> Tuple[List[Any], List[Any]]:
        pass

    
    @abstractmethod
    def dataset_y_classification_num(self, dataset_y: List[Any]) -> List[int]:
        pass
    
    
    @abstractmethod
    def get_save_file_extension(self) -> str:
        pass


    def get_best_model(self, models: List[BaseMLModel], performances: List[ModelPerformance]) -> BaseMLModel:
        index = 0
        max_acc = 0
        
        for i in range(len(performances)):
            if performances[i].get_acc() > max_acc:
                max_acc = performances[i].get_acc()
                index = i
                
        return models[index]
    
    
    def get_avg_performance(self, performances: List[ModelPerformance]) -> ModelPerformance:
        return ModelPerformance(
            acc=sum(map(lambda x: x.get_acc(), performances)) / len(performances),
            pre=sum(map(lambda x: x.get_pre(), performances)) / len(performances),
            rec=sum(map(lambda x: x.get_rec(), performances)) / len(performances),
            f1=sum(map(lambda x: x.get_f1(), performances)) / len(performances),
            roc=sum(map(lambda x: x.get_roc(), performances)) / len(performances)
        )


    def replace_data(self, dataset_x: List[Any], dataset_y: List[Any]) -> Tuple[List[int], List[int]]:
        output_x = [0] * len(dataset_x)
        output_y = self.dataset_y_classification_num(dataset_y)
        
        for i in range(len(output_x)):
            output_x[i] = i
        
        return output_x, output_y


    def get_data_split(self, dataset_x: List[Any], dataset_y: List[Any]) -> List[DatasetSplit]:
        replaced_x, replaced_y = self.replace_data(dataset_x, dataset_y)
        skf1 = StratifiedKFold(n_splits=BaseMLModel.NUM_SPLIT, shuffle=True, random_state=BaseMLModel.SEED1)
        output = []
                
        for train_val, test in skf1.split(replaced_x, replaced_y):
            skf2 = StratifiedKFold(n_splits=BaseMLModel.NUM_SPLIT, shuffle=True, random_state=BaseMLModel.SEED2)
            curr_split = DatasetSplit()
            test_x = [None] * len(test)
            test_y = [None] * len(test)
            train_val_x = [None] * len(train_val)
            train_val_y = [None] * len(train_val)
            
            for i in range(len(test)):
                test_x[i] = dataset_x[test[i]]
                test_y[i] = dataset_y[test[i]]
                            
            curr_split.set_test_set((test_x, test_y))
                
            for i in range(len(train_val)):
                train_val_x[i] = replaced_x[train_val[i]]
                train_val_y[i] = replaced_y[train_val[i]]
                                            
            for train, val in skf2.split(train_val_x, train_val_y):
                train_x = [None] * len(train)
                train_y = [None] * len(train)
                val_x = [None] * len(val)
                val_y = [None] * len(val)
                
                for i in range(len(val)):
                    val_x[i] = dataset_x[train_val_x[val[i]]]
                    val_y[i] = dataset_y[train_val_x[val[i]]]
                    
                for i in range(len(train)):
                    train_x[i] = dataset_x[train_val_x[train[i]]]
                    train_y[i] = dataset_y[train_val_x[train[i]]]
                    
                curr_split.add_validation_set((val_x, val_y))
                curr_split.add_train_set((train_x, train_y))
                
            output.append(curr_split)
            
        return output
    
    
    def k_fold(self, dataset: List[PatientData]) -> ModelPerformance:
        dataset_x, dataset_y = self.reshape_input(dataset)
        data_split = self.get_data_split(dataset_x, dataset_y)
        outer_models = [None] * len(data_split)
        outer_performances = [None] * len(data_split)
        output = None
        
        for i in range(len(data_split)):
            test_x, test_y = data_split[i].get_test_set()
            train_sets = data_split[i].get_train_sets()
            val_sets = data_split[i].get_validation_sets()
            performances = [None] * len(train_sets)
            models = [None] * len(train_sets)
            
            for j in range(len(train_sets)):
                current_model = self.create_model_copy()
                train_x, train_y = train_sets[j]
                val_x, val_y = val_sets[j]
                models[j] = current_model
                current_model.train_model_aux(train_x, train_y)
                pred_y = list(map(lambda x: x[0], current_model.predict_result_aux(val_x)))
                performances[j] = ModelPerformance.generate_performance(self.dataset_y_classification_num(val_y), pred_y)
            
            outer_models[i] = self.get_best_model(models, performances)
            pred_y = list(map(lambda x: x[0], outer_models[i].predict_result_aux(test_x)))
            outer_performances[i] = ModelPerformance.generate_performance(self.dataset_y_classification_num(test_y), pred_y)
            
        output = self.get_avg_performance(outer_performances)
            
        self.save_k_fold(outer_models, outer_performances)
            
        return output
    
    
    def save_k_fold(self, models: List[BaseMLModel], performances: List[ModelPerformance]) -> None:
        if self.is_save_k_fold != self.SAVE_MODE.NONE:
            csv_header = ["Acc", "Pre", "Rec", "F1", "ROC"]
            csv_content = []
            os.makedirs(self.save_folder, exist_ok=True)
            
            if self.is_save_k_fold == self.SAVE_MODE.BEST:
                acc_arr = [0] * len(performances)
                index = 0
                current_row = []
                best_performance = None
                for i in range(len(performances)):
                    acc_arr[i] = performances[i].get_acc()
                    
                index = np.argmax(acc_arr)
                best_performance = performances[index].get_performance()
                models[index].save_model(f"{self.save_folder}/model_{index}.{self.get_save_file_extension()}")
                for header in csv_header:
                    current_row.append(str(best_performance[header]))
                csv_content.append(current_row)
                
            elif self.is_save_k_fold == self.SAVE_MODE.ALL:
                for i in range(len(models)):
                    current_row = []
                    best_performance = performances[i].get_performance()
                    for header in csv_header:
                        current_row.append(str(best_performance[header]))
                    models[i].save_model(f"{self.save_folder}/model_{i}.{self.get_save_file_extension()}")
                    csv_content.append(current_row)
                    
            with open(f"{self.save_folder}/performance.csv", "w", encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(csv_header)
                csv_writer.writerows(csv_content)
                

    
    
    def set_save_k_fold(self, is_save_k_fold: BaseMLModel.SAVE_MODE, save_folder: str = None) -> None:
        self.is_save_k_fold = is_save_k_fold
        self.save_folder = save_folder if save_folder is not None else self.save_folder
                            
            

if __name__ == "__main__":
    pass