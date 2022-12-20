# Millimeter-Wave Beamtracking in the COSMOS Testbed Using Analog AI Accelerators
---
* Panagiotis Skrimponis, NYU Tandon School of Engineering
---

## Abstract &#x1F4D8;
Wireless communications over millimeter wave (mmWave) bands rely on narrow electrically steerable beams to overcome blockages and pathloss. In scenarios with user mobility, it is necessary to develop efficient solutions to track the available beams and find the best option. Therefore, latency and energy efficiency are very critical for this application. In this work, we explore the use of the IBM Analog AI hardware accelerators [1] as our beamtracking solution. For this application we explore the accuracy of the beamtracking model on various analog devices (e.g., PCM, RRAM, ECRAM). Most of the works in the literature rely on network simulations to generate data for beamtracking. In this work we use COSMOS [2], an advanced wireless testbed deployed in NYC, to generate the beamtracking data. On top of these data we fit a mobility model for a specific scenario (e.g., a human playing a virtual reality (VR) game [3])

## Dataset is open source :arrow_down:
The dataset can be found [here]()

## Training code :open_file_folder:
Our training code, can be found [here]().

## Main results :mortar_board:
> ![Results](https://raw.githubusercontent.com/skrimpon/nonlin/main/performance_eval.png)
>The performance of our DNN compared with the state of the art baseline. We observe a performance improvement up to 20 dB compared to the baseline. More importantly, our model significantly improves the performance specifically in the non-linear region.

## References
1. M. J. Rasch, D. Moreda, T. Gokmen, M. Le Gallo, F. Carta, C. Goldberg, K. El Maghraoui, A. Sebastian, and V. Narayanan, “A flexible and fast pytorch
   toolkit for simulating training and inference on analog crossbar arrays,” in 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and
   Systems (AICAS), 2021, pp. 1–4.
2. D. Raychaudhuri, I. Seskar, G. Zussman, T. Korakis, D. Kilper, T. Chen, J. Kolodziejski, M. Sherman, Z. Kostic, X. Gu, H. Krishnaswamy, S. Maheshwari,
   P. Skrimponis, and C. Gutterman, “Challenge: COSMOS: A city-scale programmable testbed for experimentation with advanced wireless,” in Proc. ACM
   MobiCom’20, 2020.
3. E. R. Bachmann, E. Hodgson, C. Hoffbauer, and J. Messinger, “Multi-user redirected walking and resetting using artificial potential fields,” IEEE
   Transactions on Visualization and Computer Graphics, vol. 25, no. 5, pp. 2022–2031, 2019
