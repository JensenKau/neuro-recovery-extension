from __future__ import annotations
import glob
import os
from pprint import pprint
import random
import shutil

if __name__ == "__main__":
    dataset_path = r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset"
    times = ["11 hour", "13 hour"]
    random.seed(12345)
    
    final_good = []
    final_bad = []
    
    for time in times:
        current_path = os.path.join(dataset_path, time)
    
        patient_files = glob.glob(os.path.join(current_path, "*", "*.txt"))
        
        good_set = []
        bad_set = []
        
        for filename in patient_files:
            with open(filename, "r") as file:
                if "good" in file.read().lower():
                    good_set.append(filename)
                else:
                    bad_set.append(filename)
                    
        random.shuffle(good_set)
        random.shuffle(bad_set)
        
        for i in range(100):
            final_good.append(good_set[i])
            final_bad.append(bad_set[i])
            
    for name in final_good:
        time = "11" if "11 hour" in name else "13"
        shutil.copytree(os.path.dirname(name), os.path.join(dataset_path, "test", f"{os.path.basename(name).replace(".txt", "")}_{time}_good"))
        
    for name in final_bad:
        time = "11" if "11 hour" in name else "13"
        shutil.copytree(os.path.dirname(name), os.path.join(dataset_path, "test", f"{os.path.basename(name).replace(".txt", "")}_{time}_poor"))
    
    print(len(final_good), len(final_bad))