from __future__ import annotations
import os
import glob
from pprint import pprint

from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

from patientdata.eeg_data import PatientEEGData

if __name__ == "__main__":
    folder = r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train\0954"
    files = glob.glob(os.path.join(folder, "*.hea"))

    data = PatientEEGData.load_eeg_datas(files, list(map(lambda x: x.replace(".hea", ".mat"), files)))
    
    print(data.get_eeg_data()["C3"].shape)
    print(data.get_num_points())