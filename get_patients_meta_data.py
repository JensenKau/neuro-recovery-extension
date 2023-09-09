#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:00:55 2023

@author: student
"""
from tqdm import tqdm
import os
import pandas as pd

# Find the folders with data files. (Copied from helper_code.py)
def find_data_folders(root_folder):
    data_folders = list()
    for x in sorted(os.listdir(root_folder)):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_file = os.path.join(data_folder, x + '.txt')
            if os.path.isfile(data_file):
                data_folders.append(x)
    return sorted(data_folders)

# Load the patient metadata: age, sex, etc. (Copied from helper_code.py)
def load_meta_data(data_folder, patient_id):
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    patient_metadata = load_text_file(patient_metadata_file)
    return patient_metadata

# Load text file as a dictionary. (modified from helper_code.py)
def load_text_file(filename):
    # Step 1: Read the Text File
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Step 2: Parse the Text Data
    data_dict = {}
    for line in lines:
        key, value = line.strip().split(': ')
        data_dict[key] = value
    
    return data_dict

if __name__ == '__main__':
    # make sure to edit the correct path of the database 
    data_folder = '/media/student/HDD 21/FYP_2023/physionet.org/files/i-care/2.0/training'
    
    patient_ids = find_data_folders(data_folder) # get list of patient IDs 
    num_patients = len(patient_ids) # Total number of patients
    
    # create empty dataframe to store the meta data
    df = pd.DataFrame(columns=['Patient', 'Hospital', 'Age', 'Sex', 'ROSC', 
                               'OHCA','Shockable Rhythm', 'TTM', 'Outcome', 'CPC'])
    
    # iterate over the patients' folders, read their meta data and update the dataframe
    for i in tqdm(range(num_patients)):
       patient_metadata = load_meta_data(data_folder, patient_ids[i])
       df.loc[i] = patient_metadata 
       
    # save the dataframe as excel document 
    df.to_excel('challenge_data_reference.xlsx', index=False)
    
    outcome_good = df.loc[df['Outcome']=='Good']
    outcome_good.to_excel('Good_Outcome_data.xlsx', index=False)
    
    CPC_1 = outcome_good.loc[outcome_good['CPC']=='1']