from __future__ import annotations
from typing import Tuple
from numpy.typing import NDArray
from enum import Enum

import scipy.io
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from utility.clean_eeg_data import *
from default.helper_code import load_recording_data, get_utility_frequency
import shutil
from tqdm import tqdm

class DataProcessing:
    def __init__(self, data: NDArray, sampling_frequency: int, utility_frequency: int, is_preprocessed: bool = False) -> None:
        self.data = data if is_preprocessed else self.preprocess_data(data, sampling_frequency, utility_frequency)
        
    
    def preprocess_data(self, data: NDArray, sampling_frequency: int, utility_frequency: int) -> Tuple[NDArray, int]:
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

if __name__ == "__main__":
    load_dotenv()
    
    data_folder = os.getenv("TRAINING_FOLDER")
    cleaned_folder = os.getenv("CLEANED_FOLDER")

    patient_ids = find_data_folders(data_folder)
    
    for i in tqdm(range(len(patient_ids))):
        current_dir = f"{data_folder}/{patient_ids[i]}"
        cleaned_dir = f"{cleaned_folder}/{patient_ids[i]}"
        files = sorted(set(map(lambda x: x.replace(".hea", "").replace(".mat", ""), os.listdir(current_dir))))
        
        if not os.path.exists(cleaned_dir):
            os.makedirs(cleaned_dir)
                
        for file in files:
            if file.endswith(".txt"):
                if not os.path.isfile(f"{cleaned_dir}/{file}"):
                    shutil.copyfile(f"{current_dir}/{file}", f"{cleaned_dir}/{file}")
            else:
                if not os.path.isfile(f"{cleaned_dir}/{file}.mat") or not os.path.isfile(f"{cleaned_dir}/{file}.hea"):
                    data, channels, sampling_frequency = load_recording_data(f"{current_dir}/{file}")
                    utility_frequency = None
                    
                    with open(f"{current_dir}/{file}.hea", "r", encoding="utf-8") as header_file:
                        utility_frequency = int(header_file.read().strip().split("\n")[-3].split(": ")[1])
                        
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    
                    scipy.io.savemat(f"{cleaned_dir}/{file}.mat", {"val": data})
                    
                    shutil.copyfile(f"{current_dir}/{file}.hea", f"{cleaned_dir}/{file}.hea")
                
                    
    # data = scipy.io.loadmat(f"{cleaned_folder}/0284/0284_001_004_EEG.mat")["val"]
        
    # data, channels, sampling_frequency = load_recording_data(f"{data_folder}/0284/0284_001_004_EEG")
    # utility_frequency = 50
    # data, sampling_frequency = preprocess_data(data, sampling_frequency, 50)
    
    # plt.plot(range(len(data[0])), data[0])
    # plt.show()
    
    # print(data)
    # print(len(data), len(data[0]))
    
    # print(data[0])
    # print(utility_frequency)
    # print(channels)
    # print(sampling_frequency)