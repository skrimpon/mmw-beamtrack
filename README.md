# Millimeter-Wave Beamtracking in the COSMOS Testbed Using Analog AI Accelerators
---
* Panagiotis Skrimponis, NYU Tandon School of Engineering
---

## Abstract &#x1F4D8;
Wireless communications over millimeter wave (mmWave) bands rely on narrow electrically steerable beams to overcome blockages and pathloss. In scenarios with user mobility, it is necessary to develop efficient solutions to track the available beams and find the best option. Therefore, latency and energy efficiency are very critical for this application. In this work, we explore the use of the IBM Analog AI hardware accelerators [1] as our beamtracking solution. For this application we explore the accuracy of the beamtracking model on various analog devices (e.g., PCM, RRAM, ECRAM). Most of the works in the literature rely on network simulations to generate data for beamtracking. In this work we use COSMOS [2], an advanced wireless testbed deployed in NYC, to generate the beamtracking data. On top of these data we fit a mobility model for a specific scenario (e.g., a human playing a virtual reality (VR) game [3])

## Dataset is open source :arrow_down:
The open-source dataset is located [here](https://drive.google.com/file/d/1J3RXL1FtX_H-Bjax4-G_3GRF6OosaxbD/view?usp=share_link).

## Project directory :open_file_folder:
Using the code [here](https://github.com/skrimpon/mmw-beamtrack/tree/main/project) to generate the dataset, train and test the reference and analog AI solutions.

## Python environment :snake:

We use conda to create the Python virtual environment. To regenerate the environment perform the following. The code was tested
in a Linux environment. This is a requirement if you want to use the IBM AI hardware toolkit with a GPU.

```shell
> conda env create --file=environment.yaml
```

## Regenerating the results :arrow_forward:
Follow the detailed description [here](https://github.com/skrimpon/mmw-beamtrack/tree/main/project/README.md).


## Main results :mortar_board:
> We compare the performance of analog AI accelerators with a PyTorch based LSTM solutions.
> 
> ![Results](https://raw.githubusercontent.com/skrimpon/mmw-beamtrack/main/performance_eval.png)
>
> Even though we track only only 4 beams out of the total 17 beam, we observe that all solutions achieve <2 dB misalignment loss 90% of the time. 

## References
1. M. J. Rasch, D. Moreda, T. Gokmen, M. Le Gallo, F. Carta, C. Goldberg, K. El Maghraoui, A. Sebastian, and V. Narayanan, “A flexible and fast pytorch
   toolkit for simulating training and inference on analog crossbar arrays,” in 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and
   Systems (AICAS), 2021, pp. 1–4.
2. D. Raychaudhuri, I. Seskar, G. Zussman, T. Korakis, D. Kilper, T. Chen, J. Kolodziejski, M. Sherman, Z. Kostic, X. Gu, H. Krishnaswamy, S. Maheshwari,
   P. Skrimponis, and C. Gutterman, “Challenge: COSMOS: A city-scale programmable testbed for experimentation with advanced wireless,” in Proc. ACM
   MobiCom’20, 2020.
3. E. R. Bachmann, E. Hodgson, C. Hoffbauer, and J. Messinger, “Multi-user redirected walking and resetting using artificial potential fields,” IEEE
   Transactions on Visualization and Computer Graphics, vol. 25, no. 5, pp. 2022–2031, 2019
