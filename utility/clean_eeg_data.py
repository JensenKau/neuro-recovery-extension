#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:20:44 2023

@author: student
"""
from tqdm import tqdm
import os
import pandas as pd
from default.helper_code import *
import mne

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

def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency


if __name__ == '__main__':
    # make sure to edit the correct path of the database 
    data_folder = '/media/student/HDD 21/FYP_2023/physionet.org/files/i-care/2.0/training'
    
    patient_ids = find_data_folders(data_folder) # get list of patient IDs 
    num_patients = len(patient_ids) # Total number of patients
    
    all_data = []
    # iterate over the patients' folders, read their meta data and update the dataframe
    for i in tqdm(range(num_patients)):
        recording_ids = find_recording_files(data_folder, patient_ids[i])
        recording_id = recording_ids[-1] # select the last recording in the patient folder.
        recording_location = os.path.join(data_folder, patient_ids[i], '{}_{}'.format(recording_ids[i], 'EEG'))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency, utility_frequency = load_recording_data(recording_location) # read the header and raw-eeg data
            # utility_frequency = get_utility_frequency(recording_location + '.hea') 

            # data, channels = reduce_channels(data, channels, eeg_channels)
            # you can visualize the cleaning/filtering step here (plot sample data before and after filtering)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
        else:
            print('This patient {[]} does not contain data files'.format(patient_ids[i]))
        
        all_data.append(data)
        