from __future__ import annotations
from typing import Dict, List, Tuple

import mne
import numpy as np
from numpy.typing import NDArray
import scipy.signal


def calibrate_sample(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | NDArray[np.float64]]]
    ) -> None:
    for _, eeg_set in eeg:
        eeg_set["EEG"] = (eeg_set["EEG"] - eeg_set["Offset"]) / eeg_set["Gain"]



def interpolate_sample(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | NDArray[np.float64]]]
    ) -> None:
    samples = meta["Number of Samples"]
    freq = meta["Sampling Frequency"]
    duration = meta["End Time"] - meta["Start Time"] + 1

    if duration * freq != samples:
        meta["Number of Samples"] = duration * freq
        samples = duration * freq

    for _, eeg_set in eeg.items():
        if len(eeg_set["EEG"]) < samples:
            eeg_set["EEG"] = np.concatenate(
                (eeg_set["EEG"], np.zeros(samples - eeg_set["EEG"].shape[0], dtype=np.float64)), 
                dtype=np.float64
            )
        elif len(eeg_set["EEG"]) > samples:
            eeg_set["EEG"] = eeg_set["EEG"][:samples]



def sample_resampling(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | NDArray[np.float64]]],
        new_frequency: int = 128
    ) -> None:
    lcm = np.lcm(meta["Sampling Frequency"], new_frequency)
    up = lcm // meta["Sampling Frequency"]
    down = lcm // new_frequency

    for _, eeg_set in eeg.items():
        eeg_set["EEG"] = scipy.signal.resample_poly(eeg_set["EEG"], up, down)

    meta["Sampling Frequency"] = new_frequency
    meta["Number of Samples"] = (meta["End Time"] - meta["Start Time"] + 1) * new_frequency



def merge_samples(
        metas: List[Dict[str, int]], 
        eegs: List[Dict[str, Dict[str, float | int | NDArray[np.float64]]]]
    ) -> Tuple[Dict[str, int], Dict[str, float | int | NDArray[np.float64]]]:
    sorted_data = sorted(zip(metas, eegs), key=lambda x: x[0]["Start Time"])
    sorted_meta, sorted_eeg = zip(*sorted_data)

    eeg_output = {}
    meta_output = {
        "Sampling Frequency": min(map(lambda x: x["Sampling Frequency"], sorted_meta)),
        "Number of Samples": (sorted_meta[0]["Start Time"] - sorted_meta[-1]["End Time"] + 1) * meta_output["Sampling Frequency"],
        "Utility Frequency": sorted_meta[0]["Utility Frequency"],
        "Start Time": sorted_meta[0]["Start Time"],
        "End Time": sorted_meta[-1]["End Time"]
    }

    for i in range(len(sorted_meta)):
        for channel, eeg_set in sorted_eeg[i].items():
            if channel not in eeg_output:
                eeg_output[channel] = {"EEG": [eeg_set["EEG"]]}
            else:
                prev_end = sorted_meta[i - 1]["End Time"]
                curr_start = sorted_meta[i]["Start Time"]
                eeg_output[channel]["EEG"].append(np.zeros(curr_start - prev_end - 1))
                eeg_output[channel]["EEG"].append(eeg_set["EEG"])

    for channel, eeg_set in eeg_output:
        eeg_output[channel]["EEG"] = np.concatenate(eeg_output[channel]["EEG"], dtype=np.float64)

    return meta_output, eeg_output


def denoise_sample(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | NDArray[np.float64]]],
        low_pass: float = 0.1,
        high_pass: float = 30.0
    ) -> None:
    sampling_freq = meta["Sampling Frequency"]
    utility_freq = meta["Utility Frequency"]
    for _, eeg_set in eeg:
        if low_pass <= utility_freq <= high_pass:
            eeg_set["EEG"] = mne.filter.notch_filter(eeg_set["EEG"], sampling_freq, utility_freq, n_jobs=4, verbose="error")
        eeg_set["EEG"] = mne.filter.filter_data(eeg_set["EEG"], sampling_freq, low_pass, high_pass, n_jobs=4, verbose="error")



if __name__ == "__main__":
    pass