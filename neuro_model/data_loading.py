from __future__ import annotations
import glob
import os

from tqdm import tqdm

from s3_access import S3Client

if __name__ == "__main__":
    client = S3Client("neuro-raw")

    for file in tqdm(glob.glob(r"E:\Projects\Coma Prediction\Raw Dataset\12 hour\*\*")):
        client.upload_file(file, "/".join(file.split(os.sep)[-3:]).replace("12 hour", "12-hour"))
