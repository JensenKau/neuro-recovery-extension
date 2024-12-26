from __future__ import annotations
from typing import BinaryIO, Dict, List, Tuple

import scipy.io as sio
import numpy as np
from numpy.typing import NDArray


def parse_meta(file: BinaryIO) -> Dict[str, float | str | bool]:
    def parse_value(value: str) -> float | str | bool:
        value = value.lower()

        if value == "nan": return None
        if value == "true": return True
        if value == "false": return False

        try:
            return float(value)
        except ValueError:
            return value

    output = {}

    for line in file.read().strip().split("\n"):
        key, value = line.split(": ")
        output[key] = parse_value(value)

    return output



def parse_eeg(header: BinaryIO, file: BinaryIO) -> Tuple[Dict[str, int], Dict[str, float | int | NDArray[np.float64]]]:
    def read_time(time: str) -> int:
        hour, minute, second = list(map(lambda x: int(x), time.split(":")))
        return (hour * 3600) + (minute * 60) + second

    meta_output = {}
    eeg_output = {}
    raw_header = header.read().strip().split("\n")
    raw_eeg = sio.loadmat(file)["val"]

    meta_output["Sampling Frequency"], meta_output["Number of Samples"] = list(map(lambda x: int(x), raw_header[0].split()[-2:]))
    meta_output["Utility Frequency"] = int(raw_header[-3].split()[1])
    meta_output["Start Time"] = read_time(raw_header[-2].split()[1])
    meta_output["End Time"] = read_time(raw_header[-1].split()[1])

    for i in range(len(raw_eeg)):
        line = raw_header[i + 1].split()
        eeg_output[line[8]] = {
            "Gain": float(line[2].split("/")[0]),
            "Offset": int(line[4]),
            "EEG": raw_eeg[i].astype(np.float64)
        }

    return meta_output, eeg_output



if __name__ == "__main__":
    pass