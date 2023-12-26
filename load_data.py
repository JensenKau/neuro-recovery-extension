from __future__ import annotations
from typing import Tuple, List

import os
import re
from tqdm import tqdm

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
    def get_12hr_file(cls, folder: str) -> str:
        file_list = os.listdir(folder)
        
        for file in file_list:
            if re.match(f"\d\d\d\d_\d\d\d_012_EEG\.hea", file) is not None: 
                return file[:-4]
                

if __name__ == "__main__":
    # Balanced out good/bad patient
    patients = [
        '0312', '0319', '0326', '0334', '0335', '0341', '0347', '0348', '0353', '0354', '0355', '0357', '0359', '0362', '0369', 
        '0377', '0378', '0380', '0384', '0389', '0394', '0396', '0400', '0410', '0411', '0415', '0420', '0421', '0428', '0429', 
        '0432', '0433', '0438', '0445', '0448', '0450', '0451', '0459', '0460', '0462', '0463', '0464', '0465', '0466', '0468', 
        '0471', '0472', '0477', '0481', '0500', '0514', '0518', '0532', '0541', '0544', '0547', '0555', '0564', '0566', '0571', 
        '0574', '0577', '0582', '0584', '0587', '0592', '0595', '0602', '0604', '0606', '0614', '0616', '0624', '0627', '0628', 
        '0631', '0639', '0642', '0644', '0645', '0646', '0647', '0649', '0650', '0651', '0652', '0655', '0663', '0665', '0668', 
        '0671', '0673', '0678', '0680', '0681', '0684', '0685', '0688', '0689', '0692', '0700', '0703', '0709', '0710', '0715', 
        '0728', '0741', '0742', '0744', '0746', '0748', '0749', '0752', '0754', '0757', '0758', '0764', '0765', '0768', '0774', 
        '0777', '0780', '0787', '0788', '0796', '0797', '0799', '0800', '0804', '0806', '0807', '0808', '0809', '0810', '0811', 
        '0814', '0816', '0820', '0822', '0827', '0832', '0834', '0837', '0839', '0840', '0847', '0860', '0862', '0870', '0872', 
        '0897', '0904', '0913', '0920', '0925', '0929', '0935', '0944', '0952', '0980'
    ]

    patient_datas = [None] * len(patients)
    patient_dataset = None
    
    for i in tqdm(range(len(patients))):
        folder = f"data/training/training/{patients[i]}"
        meta_file = f"{folder}/{LoadTrainingUtility.get_txt_file(folder)}"
        data_filename = f"{folder}/{LoadTrainingUtility.get_12hr_file(folder)}"
        
        patient_datas[i] = PatientData.load_patient_data(meta_file, f"{data_filename}.hea", f"{data_filename}.mat")
            
    patient_dataset = PatientDataset(patient_datas)
    
    print("Saving Data...")
    patient_dataset.save_dataset("balanced_connectivity.pkl")
    print("Complete!!!")