from __future__ import annotations
from typing import Dict, List, Tuple
import warnings

from nilearn.connectome import ConnectivityMeasure
import numpy as np
from numpy.typing import NDArray


def apply_normalisation(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | List[float]]]
    ) -> None:
    pass


def combine_eeg(
        meta: Dict[str, int], 
        eeg: Dict[str, Dict[str, float | int | NDArray[np.float64]]],
        order: List[str] = [
            "Fp1", "Fp2", "F7", "F8", "F3", "F4", 
            "T3", "T4", "C3", "C4", "T5", "T6", 
            "P3", "P4", "O1", "O2", "Fz", "Cz", 
            "Pz", "Fpz", "Oz", "F9"
        ]
    ) -> Tuple[Dict[str, int], NDArray[np.float64]]:
    meta["Order"] = order
    combined_eeg = []

    for channel in order:
        if channel not in eeg:
            combined_eeg.append(np.zeros(meta["Number of Samples"]))
        else:
            combined_eeg.append(eeg[channel]["EEG"])

    combined_eeg = np.array(combined_eeg)

    return combined_eeg



def static_connectivity(
        meta: Dict[str, int | List[str]], 
        eeg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    measure = ConnectivityMeasure(kind="correlation")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return measure.fit_transform(np.array([eeg]).swapaxes(1, 2))[0]
    


def dynamic_connectivity(
        meta: Dict[str, int | List[str]], 
        eeg: NDArray[np.float64],
        duration: int = 60,
        shift: float = 0.1,
        chunk: int = 1
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    window_size = meta["Sampling Frequency"] * duration
    shift_size = int(window_size * shift)
    measure = ConnectivityMeasure(kind="correlation")
    output = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(0, meta["Number of Samples"], shift_size):
            window_end = min(i + shift_size, meta["Number of Samples"])
            transformed_eeg = np.array([eeg[:, window_end]]).swapaxes(1, 2)
            current = measure.fit_transform(transformed_eeg)[0]

            if not np.isnan(current).any():
                output.append(current)
            else:
                output.append(np.zeros((meta["Order"], meta["Order"]), dtype=np.float64))

    return np.mean(output, axis=0), np.std(output, axis=0)


if __name__ == "__main__":
    pass