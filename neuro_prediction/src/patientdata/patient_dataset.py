from __future__ import annotations
from typing import List, Tuple
from numpy.typing import NDArray

import os
import dill
from src.patientdata.patient_data import PatientData

class PatientDataset:
    def __init__(self, dataset: List[PatientData]) -> None:
        self.dataset = dataset


    @classmethod
    def load_raw_dataset(cls, root_folder: str) -> PatientDataset:
        dataset = []
        
        for folder in os.listdir(root_folder):
            current_path = os.path.join(root_folder, folder)
            meta_file = None
            eeg_files = {}
            
            for file in os.listdir(current_path):
                if file.endswith(".txt"):
                    meta_file = os.path.join(current_path, file)  
                else:
                    if file.endswith(".hea"):
                        eeg_files[os.path.join(current_path, file)] = os.path.join(current_path, file.replace(".hea", ".mat"))
                    else:
                        eeg_files[os.path.join(current_path, file.replace(".mat", ".hea"))] = os.path.join(current_path, file)
                        
            for header, content in eeg_files.items():
                dataset.append(PatientData.load_patient_data(meta_file, header, content))
                
        return PatientDataset(dataset)
    
    
    @classmethod
    def load_processed_dataset(cls, file: str) -> PatientDataset:
        dataset = None
        
        with open(file, "rb") as dill_file:
            dataset = dill.load(dill_file)
        
        return PatientDataset(dataset)
    
    
    def save_dataset(self, filename: str) -> None:
        with open(filename, "wb") as dill_file:
            dill.dump(self.dataset, dill_file)
            
            
    def get_dataset(self) -> List[PatientData]: 
        return self.dataset
        

if __name__ == "__main__":
    pass