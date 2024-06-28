from __future__ import annotations
import os
import glob
from pprint import pprint
import random

from tqdm import tqdm
from braindecode.augmentation import FTSurrogate
import torch
import numpy as np

from src.patientdata.eeg_data import PatientEEGData

class TransformDataset:
    @classmethod
    def generate_merged_data(cls, path: str) -> None:
        traversal = list(os.walk(path))
        for i in tqdm(range(len(traversal))):
            filenames = traversal[i][2]
            if len(filenames) > 0:                
                headers = list(filter(lambda x: ".hea" in x, filenames))
                contents = list(filter(lambda x: ".mat" in x, filenames))
                                
                for j in range(len(headers)):
                    headers[j] = os.path.join(traversal[i][0], headers[j])
                for j in range(len(contents)):
                    contents[j] = os.path.join(traversal[i][0], contents[j])
                
                data = PatientEEGData.load_eeg_datas(headers, contents)
                data.save_eeg(os.path.join(os.path.dirname(headers[0]), "merged.mat"))
                
    
    @classmethod
    def delete_raw_data(cls, path: str) -> None:
        traversal = list(os.walk(path))
        for i in range(len(traversal)):
            folder = traversal[i][0]
            filenames = traversal[i][2]
            if len(filenames) > 0:
                files = filter(lambda x: ".hea" in x or (".mat" in x and "merged.mat" not in x), filenames)
                files = list(map(lambda x: os.path.join(folder, x), files))
                for file in files:
                    os.remove(file)
    
    
    @classmethod
    def apply_ft_surrogate(cls, path: str) -> None:
        traversal = list(os.walk(path))
        transform = FTSurrogate(1, random_state=123)
        random.seed(12345)
        
        for i in tqdm(range(len(traversal))):
            folder = traversal[i][0]
            filenames = traversal[i][2]
            if folder != path and folder.endswith("_aug"):
                mat = list(filter(lambda x: x == "merged.mat", filenames))[0]
                eeg = PatientEEGData.load_processed_eeg(os.path.join(folder, mat))
                regions = list(eeg.get_eeg_data().keys())
                
                random.shuffle(regions)
                
                for i in range(len(regions) // 2):
                    region = regions[i]
                    augmented = transform.operation(torch.from_numpy(np.array([eeg.get_eeg_data()[region]])), None, 1, False, 123)[0]
                    augmented = np.float64(augmented.numpy()[0][0])
                    eeg.set_eeg_data(region, augmented)
                    
                eeg.save_eeg(os.path.join(folder, "merged.mat"))
        

if __name__ == "__main__":
    pprint(list(os.walk(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset")))