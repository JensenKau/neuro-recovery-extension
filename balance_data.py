from __future__ import annotations
from typing import Tuple, List

import os
import re

from patientdata.patient_data import PatientData
from patientdata.patient_dataset import PatientDataset

class LoadTrainingUtility:
    @classmethod
    def get_txt_file(cls, folder: str) -> str:
        file_list = os.listdir(folder)
        
        for file in file_list:
            if file.endswith(".txt"):
                return file
    
    
    @classmethod
    def get_12hr_file(cls, folder: str) -> List[str]:
        output = []
        file_list = os.listdir(folder)
        
        for file in file_list:
            if re.match(f"\d\d\d\d_\d\d\d_012_EEG\.hea", file) is not None: 
                output.append(file[:-4])
                
        return output


if __name__ == "__main__":
    patients = None
    
    with open("patient.txt", "r", encoding="utf-8") as file:
        patients = file.read().split("\n")
    
    good = 0
    bad = 0
    actual = [None] * 160
    pointer = 0
    
    for i in range(len(patients)):
        folder = f"data/training/training/{patients[i]}"
        meta_file = LoadTrainingUtility.get_txt_file(folder)
        data_file = LoadTrainingUtility.get_12hr_file(folder)
        
        with open(f"{folder}/{meta_file}", "r", encoding="utf-8") as file:
            is_good = "Good" in file.read()
            
            if is_good and good < 80 and len(data_file) == 1:
                actual[pointer] = patients[i]
                good += 1
                pointer += 1

            elif not is_good and bad < 80 and len(data_file) == 1:
                actual[pointer] = patients[i]
                bad += 1
                pointer += 1
                
    print(actual)