from __future__ import annotations
from typing import Tuple

from numpy.typing import NDArray
import numpy as np
import scipy
import mne
from nilearn.connectome import ConnectivityMeasure



class EegProcessing:
    @classmethod
    def resample_eeg(cls, data: NDArray, sampling_frequency: int) -> Tuple[NDArray, int]:
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
    pass