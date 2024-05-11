from __future__ import annotations

import numpy as np
import torch
from nilearn import plotting
import matplotlib.pyplot as plt
from PIL import Image

from src.patientdata.patient_data import PatientData
from src.patientdata.patient_dataset import PatientDataset
from src.patientdata.connectivity_data import PatientConnectivityData
from src.load_data import load_data

if __name__ == "__main__":
    patient_dataset = PatientDataset.load_processed_dataset("src/balanced_connectivity.pkl")
    data = patient_dataset.get_dataset()[0].get_static_fc()
    
    # view = plotting.view_connectome(
    #     data, coords, edge_threshold="80%"
    # )
    
    test = plotting.plot_matrix(data, labels=PatientConnectivityData.BRAIN_REGION, colorbar=True, vmax=1, vmin=1).make_image("agg")
    res = Image.fromarray(test[0])
    res.save("bcd.png")