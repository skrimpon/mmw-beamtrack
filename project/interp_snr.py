"""
@organization: New York University
@author: Panagiotis Skrimponis
@contact: ps3857@nyu.edu

@copyright: 2022
"""
import numpy as np
import pandas as pd
from glob import glob
from scipy import interpolate
from datetime import datetime



def main(cosmos_file, traj_dir):
    # Read COSMOS measurements
    df = pd.read_csv(cosmos_file)

    x = np.array(df['Loc_X'], dtype=float)
    y = np.array(df['Loc_Y'], dtype=float)
    angle = np.array(df['Angle'], dtype=float)
    beam_index = np.array(df['Beam_Index'], dtype=int)
    snr = np.array(df['SNR'], dtype=float)

    x_test = np.unique(x)
    y_test = np.unique(y)
    beam_test = np.unique(beam_index)
    angle_test = np.unique(angle)

    gamma_a0_b0 = np.zeros((len(beam_test), len(angle_test), len(x_test), len(y_test)))
    for l, bi in enumerate(beam_test):
        for k, ai in enumerate(angle_test):
            for i, xi in enumerate(x_test):
                for j, yi in enumerate(y_test):
                    gamma_a0_b0[l, k, i, j] = snr[(x == xi) & (y == yi) & (beam_index == bi) & (angle == ai)]

    files = glob(traj_dir+"*.npy")
    for file_idx, f in enumerate(files):
        if file_idx < 2375:
            continue
        f = f.replace('\\', '/')
        fname = f.split('/')[-1].split('.')[0]
        df = pd.DataFrame(columns=['Loc_X', 'Loc_Y', 'Angle', 'Beam_Index', 'SNR'])
        data = np.load(f, allow_pickle=True)
        isamp = 0
        t0 = datetime.now()
        print(f'\nProcessing {file_idx + 1}/{len(files)}...')
        for d in data:
            for i, bi in enumerate(beam_test):
                f = interpolate.interp2d(x_test, x_test, gamma_a0_b0[i, angle_test == d[2]], kind='linear')
                df.loc[isamp] = [d[0], d[1], d[2], i, f(d[0], d[1])[0]]
                isamp += 1
                print(f'\rElapsed: {datetime.now() - t0}, Progress: {isamp / len(data) / len(beam_test) * 100:.2f} %',
                      end='')
        df.to_csv(f"../data/dataset/{fname}.csv", index=False)
        print('')

if __name__ == "__main__":
    cosmos_file = "../data/202211301204_beamtracking.csv"
    traj_dir = "../data/traj/"
    try:
        main(cosmos_file=cosmos_file, traj_dir=traj_dir)
    except Exception as e:
        print(f"ERROR: {e}")
