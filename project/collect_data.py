"""
:organization: New York University
:author: Panagiotis Skrimponis

:copyright: 2022
"""
# Import Libraries
import os
import sys
import time
import socket
import argparse
import configparser
import pandas as pd
import numpy as np
from datetime import datetime

path = os.path.abspath('../../mmwsdr/host/')
if not path in sys.path:
    sys.path.append(path)
import mmwsdr


def main():
    """
    Main function
    """

    # Parameters
    nfft = 1024  # num of continuous samples per frames
    nskip = 0  # num of samples to skip between frames
    nframe = 8  # num of frames
    isdebug = False  # print debug messages
    iscalibrated = True  # apply calibration parameters
    sc_min = -400  # min sub-carrier index
    sc_max = 400  # max sub-carrier index
    tx_pwr = 12000  # transmit power

    # Create the test vectors
    x_test = np.linspace(0, 1300, 14, dtype=int)
    y_test = np.linspace(0, 1300, 14, dtype=int)
    angle_test = np.linspace(-45, 45, 7)
    beam_test = np.array([1, 5, 9, 13, 17, 21, 25, 29, 32, 35, 39, 43, 47, 51, 55, 59, 63])
    nsamp = len(x_test) * len(y_test) * len(angle_test) * len(beam_test)

    node = socket.gethostname().split('.')[0]  # Find the local hostname

    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=float, default=60.48e9, help="Carrier frequency in Hz (i.e., 60.48e9)")
    args = parser.parse_args()

    # Create a configuration parser
    config = configparser.ConfigParser()
    config.read('../../config/sivers.ini')

    # Create the SDR objects and the XY table controllers
    sdr1 = mmwsdr.sdr.Sivers60GHz(config=config, node='srv1-in1', freq=args.freq, isdebug=isdebug, islocal=(node == 'srv1-in1'), iscalibrated=iscalibrated)
    xytable1 = mmwsdr.utils.XYTable(config['srv1-in1']['table_name'], isdebug=isdebug)

    sdr2 = mmwsdr.sdr.Sivers60GHz(config=config, node='srv1-in2', freq=args.freq, isdebug=isdebug, islocal=(node == 'srv1-in2'), iscalibrated=iscalibrated)
    xytable2 = mmwsdr.utils.XYTable(config['srv1-in2']['table_name'], isdebug=isdebug)

    # Create a wide-band tx signal
    txtd = mmwsdr.utils.waveform.wideband(sc_min=sc_min, sc_max=sc_max, nfft=nfft)

    # Send the transmit sequence with cyclic repeat
    sdr1.send(txtd * tx_pwr)

    # Move TX at the center facing at 0 deg
    xytable1.move(x=650, y=650, angle=0)

    # Main loop
    data = pd.DataFrame(columns=['Loc_X', 'Loc_Y', 'Angle', 'Beam_Index', 'SNR'])
    isamp = 0
    for x in x_test:
        for y in y_test:
            for angle in angle_test:
                xytable2.move(x, y, angle)
                time.sleep(1)
                for beam_index in beam_test:
                    sdr2.beam_index = beam_index
                    time.sleep(0.5)

                    # Receive data
                    rxtd = sdr2.recv(nfft, nskip, nframe)
                    rxfd = np.fft.fft(rxtd, axis=1)
                    Hest = rxfd * np.conj(np.fft.fft(txtd))
                    hest = np.fft.ifft(Hest, axis=1)
                    pdp = 20 * np.log10(np.abs(hest))
                    sig_max = np.max(pdp, axis=1)
                    sig_avg = np.mean(pdp, axis=1)
                    snr = np.mean(sig_max - sig_avg)
                    data.loc[isamp].append([x, y, angle, beam_index, snr])
                    isamp += 1
                    print(f"\rElapsed Time: {datetime.now()}, Progress: {isamp/nsamp:.2f}%", end='')
                    break
                break
            break
        break
    print('')

    data.to_csv(time.strftime("%Y%m%d%H%M_beamtracking.csv"))

    # Delete the SDR object. Close the TCP connections.
    del sdr1, sdr2


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass