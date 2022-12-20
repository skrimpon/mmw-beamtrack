"""
@organization: New York University
@author: Panagiotis Skrimponis
@contact: ps3857@nyu.edu

@copyright: 2022
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

def main(files, version:int=0):
    num_files = len(files)  # Number of trajectories
    num_beams = 17  # Number of beams to track
    num_data = 1000  # Number of data points per trajectory

    data = np.zeros((num_files, num_data, num_beams))
    t0 = datetime.now()
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        snr = np.array(df["SNR"]).reshape((-1, num_beams))
        data[i] = snr
        print(f"\rElapsed Time: {datetime.now()-t0}, Progress: {(i+1)/len(files)*100:.2f} %", end="")
    np.savez_compressed(f"../data/beamtracking_dataset_v{version}.npz", data=data)


if __name__ == "__main__":
    input_dir = "../data/dataset/"
    output_dir = "../data/"
    version = 0

    files = glob(os.path.join(input_dir, "*.csv"))
    try:
        main(files=files, version=version)
    except Exception as e:
        print(f'ERROR: {e}')
