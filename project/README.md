#  Description
* **collect_data.py**: We use this file on COSMOS testbed. This file doesn't require any input arguments. This file controls two Xilinx RFSoC ZCU111 devices and two Sivers IMA 60 GHz boards. We place the receiver at the center of the XY table
* **generate_traj.py**: This function use the open-source repository pm4vr to generate random user trajectories. We generate 10,000 user trajectories in csv format. The header of the csv is position in x-axis, y-axis, and rotation along the z-axis. The file will generate 10,000 csv files under "..data/traj/". The file has been modified from the original example to map the dimensions of the XY table in COSMOS. Also, because of the scenario we consider we only need one user at a time. 
* **interp_snr.py**: After we collect the data from COSMOS and generate the user trajectories we need to fit these trajectories on COSMOS data.
* **create_dataset.py**:
* **generate_plots.py**: We use this file to generate the final plot.

# How to use
1. Follow steps in the [COSMOS tutorial](https://wiki.cosmos-lab.org/wiki/Tutorials/Wireless/mmwaveRFSoC) we make a reservation on COSMOS-sb1.
2. After we collect the measurements on COSMOS we need to generate the user trajectories.
```shell
> python generate_traj.py
```
3. To fit the COSMOS data on the user trajectories we run,
```shell
> python interp_snr.py
```
