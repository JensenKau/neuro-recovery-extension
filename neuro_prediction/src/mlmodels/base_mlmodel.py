from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
from enum import Enum
import os
import csv

import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold

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
    
    
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        
        self.k_fold_models = None
        self.k_fold_perfs = None
        self.k_fold_avg = None
        self.k_fold_std = None
        
        self.optuna_model_copy = None


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
    
    
    @abstractmethod
    def objective(self, trial: optuna.trial.Trial, dataset: List[PatientData]) -> BaseMLModel:
        pass
    
    
    def get_avg_performance(self, performances: List[ModelPerformance]) -> ModelPerformance:
        return ModelPerformance(
            acc=float(np.mean(list(map(lambda x: x.get_acc(), performances)), axis=0)),
            pre=float(np.mean(list(map(lambda x: x.get_pre(), performances)), axis=0)),
            rec=float(np.mean(list(map(lambda x: x.get_rec(), performances)), axis=0)),
            f1=float(np.mean(list(map(lambda x: x.get_f1(), performances)), axis=0)),
            roc=float(np.mean(list(map(lambda x: x.get_roc(), performances)), axis=0))
        )
        
    
    def get_stddev_performance(self, performances: List[ModelPerformance]) -> ModelPerformance:
        return ModelPerformance(
            acc=float(np.std(list(map(lambda x: x.get_acc(), performances)), axis=0)),
            pre=float(np.std(list(map(lambda x: x.get_pre(), performances)), axis=0)),
            rec=float(np.std(list(map(lambda x: x.get_rec(), performances)), axis=0)),
            f1=float(np.std(list(map(lambda x: x.get_f1(), performances)), axis=0)),
            roc=float(np.std(list(map(lambda x: x.get_roc(), performances)), axis=0))
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
    
    
    def k_fold(self, dataset: List[PatientData]) -> None:
        dataset_x, dataset_y = self.reshape_input(dataset)
        data_split = self.get_data_split(dataset_x, dataset_y)
        outer_models = [None] * len(data_split)
        outer_performances = [None] * len(data_split)
        
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
            
            outer_models[i] = models[np.argmax(list(map(lambda x: x.get_acc(), performances)))]
            pred_y = list(map(lambda x: x[0], outer_models[i].predict_result_aux(test_x)))
            outer_performances[i] = ModelPerformance.generate_performance(self.dataset_y_classification_num(test_y), pred_y)
            
        self.k_fold_models = outer_models
        self.k_fold_perfs = outer_performances
        self.k_fold_avg = self.get_avg_performance(outer_performances)
        self.k_fold_std = self.get_stddev_performance(outer_performances)
        
    
    def get_k_fold_performances(self) -> Dict[str, List[BaseMLModel] | List[ModelPerformance] | ModelPerformance]:
        return {
            "models": self.k_fold_models,
            "performances": self.k_fold_perfs,
            "avg": self.k_fold_avg,
            "std": self.k_fold_std
        }
                            
    
    def save_k_fold(self, folder: str, save_mode: SAVE_MODE = SAVE_MODE.ALL) -> None:
        if save_mode != self.SAVE_MODE.NONE:
            csv_header = ["Acc", "Pre", "Rec", "F1", "ROC"]
            csv_content = []
            os.makedirs(folder, exist_ok=True)
            
            if save_mode == self.SAVE_MODE.BEST:
                acc_arr = list(map(lambda x: x.get_acc(), self.k_fold_perfs))
                index = np.argmax(acc_arr)
                best_performance = self.k_fold_perfs[index].get_performance()
                current_row = list(map(lambda x: str(best_performance[x]), csv_header))
                self.k_fold_models[index].save_model(f"{folder}/model_{index}.{self.get_save_file_extension()}")
                csv_content.append(current_row)
                
            elif save_mode == self.SAVE_MODE.ALL:
                for i in range(len(self.k_fold_models)):
                    curr_performance = self.k_fold_perfs[i].get_performance()
                    current_row = list(map(lambda x: str(curr_performance[x]), csv_header))
                    self.k_fold_models[i].save_model(f"{folder}/model_{i}.{self.get_save_file_extension()}")
                    csv_content.append(current_row)
                    
            with open(f"{folder}/performance.csv", "w", encoding="utf-8") as file:
                csv_writer = csv.writer(file)                
                avg_row = self.k_fold_avg.get_performance()
                std_row = self.k_fold_std.get_performance()
                csv_writer.writerow(csv_header)
                csv_writer.writerows(csv_content)
                csv_writer.writerows([
                    list(map(lambda x: str(avg_row[x]), csv_header)), 
                    list(map(lambda x: str(std_row[x]), csv_header))
                ]) 
        
        
    def tune_paramters(self, iteration: int, dataset: List[PatientData]) -> None:
        study = optuna.create_study(
            study_name=self.model_name,
            storage=f"sqlite:///../../{self.model_name}.db",
            direction="maximize", 
            load_if_exists=True,
        )
        
        def model_objective(trial: optuna.trial.Trial) -> float:
            self.optuna_model_copy = self.objective(trial, dataset)
            return self.optuna_model_copy.get_k_fold_performances()["avg"].get_acc()
        
        def save_best_trial(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            if study.best_trial.number == trial.number:
                self.optuna_model_copy.save_k_fold(f"../../trained_models/{self.optuna_model_copy.model_name}")  
        
        study.optimize(
            func=model_objective, 
            n_trials=iteration,
            callbacks=[save_best_trial]
        )
            

if __name__ == "__main__":
    pass