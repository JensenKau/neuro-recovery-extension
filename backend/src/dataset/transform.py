from __future__ import annotations
import os
import glob
from pprint import pprint

from tqdm import tqdm

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
                

if __name__ == "__main__":
    pprint(list(os.walk(r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset")))