from __future__ import annotations
import glob
import os
from pprint import pprint
import random
import shutil

if __name__ == "__main__":
    dataset_path = r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset"
    dataset_path = os.path.join(dataset_path, "12 hour")
    
    patient_files = glob.glob(os.path.join(dataset_path, "*", "*.txt"))
    
    good_set = []
    bad_set = []
    
    for filename in patient_files:
        with open(filename, "r") as file:
            if "good" in file.read().lower():
                good_set.append(filename)
            else:
                bad_set.append(filename)
    
    random.seed(1234)
    random.shuffle(good_set)
    
    for i in range(len(good_set)):
        if len(good_set) < len(bad_set):
            good_set.append(os.path.join(f"{os.path.dirname(good_set[i])}_aug", os.path.basename(good_set[i])))
    
    for name in (good_set + bad_set):
        original_name = name.replace("_aug", "")
        shutil.copytree(os.path.dirname(original_name), os.path.join(dataset_path, "train", f"{os.path.basename(name).replace(".txt", "")}{'_aug' if '_aug' in name else ''}"))
            
    print(len(good_set), len(bad_set))