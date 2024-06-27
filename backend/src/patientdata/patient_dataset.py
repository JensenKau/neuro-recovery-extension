from __future__ import annotations
from typing import List
import os

import dill
from numpy.typing import NDArray

from src.patientdata.patient_data import PatientData


class PatientDataset:
    def __init__(self, dataset: List[PatientData]) -> None:
        self.dataset = dataset


    @classmethod
    def load_processed_eeg(cls, root_folder: str, keep_eeg: bool = False, keep_fc: bool = False) -> PatientDataset:
        dataset = []
        
        for folder, _, files in os.walk(root_folder):
            if folder != root_folder:
                meta = os.path.join(folder, list(filter(lambda x: x.endswith(".txt"), files))[0])
                eeg = os.path.join(folder, list(filter(lambda x: x.endswith("merged.mat"), files))[0])
                dataset.append(PatientData.load_patient_processed(meta, eeg, keep_eeg, keep_fc))
                
        return PatientDataset(dataset)
    
    
    @classmethod
    def load_raw_eeg(cls, root_folder: str, keep_eeg: bool = False, keep_fc: bool = False) -> PatientDataset:
        dataset = []
        
        for folder, _, files in os.walk(root_folder):
            if folder != root_folder:
                meta = os.path.join(folder, list(filter(lambda x: x.endswith(".txt"), files))[0])
                hea = list(map(lambda x: os.path.join(folder, x), filter(lambda x: x.endswith(".hea"), files)))
                eeg = list(map(lambda x: os.path.join(folder, x), filter(lambda x: x.endswith(".mat"), files)))
                dataset.append(PatientData.load_patient_data(meta, hea, eeg, keep_eeg, keep_fc))
        
        return PatientDataset(dataset)


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
    def load_dataset(cls, file: str) -> PatientDataset:
        dataset = None
        
        with open(file, "rb") as dill_file:
            dataset = dill.load(dill_file)
        
        return PatientDataset(dataset)
    
    
    def save_dataset(self, filename: str, add_extension: bool = True) -> None:
        if add_extension and not filename.endswith(".pkl"):
            filename = f"{filename}.pkl"
        
        with open(filename, "wb") as dill_file:
            dill.dump(self.dataset, dill_file)
            
            
    def get_dataset(self) -> List[PatientData]: 
        return self.dataset
        

if __name__ == "__main__":
    pass