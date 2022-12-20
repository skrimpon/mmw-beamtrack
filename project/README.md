#  Description :books:
* **collect_data.py**: Use this file on COSMOS testbed. This file doesn't require any input arguments. This file controls two Xilinx RFSoC ZCU111 devices and two Sivers IMA 60 GHz boards. We place the receiver at the center of the XY table
* **generate_traj.py**: This function use the open-source repository pm4vr to generate random user trajectories. We generate 10,000 user trajectories in csv format. The header of the csv is position in x-axis, y-axis, and rotation along the z-axis. The file will generate 10,000 csv files under "..data/traj/". The file has been modified from the original example to map the dimensions of the XY table in COSMOS. Also, because of the scenario we consider we only need one user at a time. 
* **interp_snr.py**: After we collect the data from COSMOS and generate the user trajectories we need to fit these trajectories on COSMOS data.
* **create_dataset.py**: Use this file to create the SNR tensor. This will combine all the data from the generated trajectories.
* **generate_plots.py**: We use this file to generate the final plot.

# Usage :arrow_forward:
1. Follow steps in the [COSMOS tutorial](https://wiki.cosmos-lab.org/wiki/Tutorials/Wireless/mmwaveRFSoC).
After you setup the FPGA at node srv1-n1:
```
root@srv1-in1:~/$ cd mmwsdr/host/mmwsdr/array
root@srv1-in1:~/mmwsdr/host/mmwsdr/array$ python ederarray.py -u SN0240
```
Then, from node srv1-in2:
```shell
root@srv1-in2:~/$ git clone git@github.com:skrimpon/mmw-beamtrack.git
root@srv1-in2:~/$ mkdir ~/mmwsdr/host/project/mmw-beamtrack
root@srv1-in2:~/$ cp mmw-beamtrack/project/collect_data.py mmwsdr/host/project/mmw-beamtrack/
root@srv1-in2:~/$ cd mmwsdr/host/project/mmw-beamtrack/
root@srv1-in2:~/mmwsdr/host/project/mmw-beamtrack/$ python the collect_data.py
```
Finally, download the dataset under ../data directory.

2. Generate the user trajectories,
```shell
$ python generate_traj.py
```

3. Fit the COSMOS data on the user trajectories we run,
```shell
$ python interp_snr.py
```

4. Create a dataset in compressed numpy format that has only the signal-to-noise ratio saved in a tensor **<number of trajectories, number of timesteps, number of beams>**
```shell
$ python create_dataset.py
```

5. Train neural networks,
```shell
$ python train_ref.py --k 4 --version 0
$ python train_analog.py --rpu 0 --k 4 --version 0
$ python train_analog.py --rpu 1 --k 4 --version 0
```

6. Run inference using the test dataset,
```shell
$ python test_ref.py --k 4 --version 0
$ python test_analog.py --rpu 0 --k 4 --version 0
$ python test_analog.py --rpu 1 --k 4 --version 0
```

7. Generate the plots,
```shell
$ python generate_plot.py
```

8. If you want to achieve better performance you can clip the maximum SNR,
```shell
$ python train_ref.py --k 4 --version 0 --clip_snr 1
$ python train_analog.py --rpu 0 --k 4 --version 0 --clip_snr 1
$ python train_analog.py --rpu 1 --k 4 --version 0 --clip_snr 1
$ python test_ref.py --k 4 --version 0 --clip_snr 1
$ python test_analog.py --rpu 0 --k 4 --version 0 --clip_snr 1
$ python test_analog.py --rpu 1 --k 4 --version 0 --clip_snr 1
```