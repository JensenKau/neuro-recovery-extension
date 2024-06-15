from __future__ import annotations
import glob
from pprint import pprint

if __name__ == "__main__":
    files = glob.glob("./dataset/13 hour/*/*.txt")
    good = 0
    bad = 0
    
    for file in files:
        with open(file, "r") as file:
            if "good" in file.read().lower():
                good += 1
            else:
                bad += 1
                
    print(f"Good: {good} | Bad: {bad}")