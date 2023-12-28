from __future__ import annotations

import numpy as np
import random
import optuna
import logging
import sys

from mlmodels.svm_model import SVMModel
from patientdata import PatientData, PatientDataset

def objective(trial: optuna.trial.Trial) -> float:
    patient_dataset = PatientDataset.load_processed_dataset("balanced_connectivity.pkl")
    
    c = trial.suggest_float("C", 0.0001, 1000)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 1, 100)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    svm = SVMModel()
    svm.initialize_model(**{"C": c, "kernel": kernel, "degree": degree, "gamma": gamma})
    
    performance = svm.k_fold(patient_dataset.get_dataset())
    
    return performance.get_acc()


if __name__ == "__main__":        
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    study_name = "svm-study"  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)
    
    study.optimize(objective, n_trials=200)
    
