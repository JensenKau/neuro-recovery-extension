import scipy.io
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from clean_eeg_data import *
from helper_code import load_recording_data, get_utility_frequency
import shutil
from tqdm import tqdm

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