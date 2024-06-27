from __future__ import annotations
import os
import glob
from pprint import pprint
import sys

import numpy as np
import matplotlib.pyplot as plt

from patientdata.eeg_data import PatientEEGData
from patientdata.connectivity_data import PatientConnectivityData

if __name__ == "__main__":
    folder = r"C:\Users\USER\Desktop\School\neuro-recovery-extension\__dataset\train\0954"
    files = glob.glob(os.path.join(folder, "*.hea"))

    data = PatientEEGData.load_eeg_datas(files, list(map(lambda x: x.replace(".hea", ".mat"), files)))
    
    fc = PatientConnectivityData.load_patient_connectivity(data)
    
    print(fc.get_static_fc())