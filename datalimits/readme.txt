📁 Data Limitation Analysis for Simultaneous Inference

This folder contains scripts for analyzing data limitations in simultaneous inference of network structure and dynamics.

1. Generate Time Series Data

Use generate_series.py (in the data/ folder) to generate time series for networks of various sizes.
Then, use calculate_deriv.py (also in data/) to compute the corresponding derivatives for each generated series.

2. Analysis Overview

The data limitation analysis is divided into three parts:

Dynamics inference under known network structure
Run dyn_learning_knstruc.py with different datasets to infer dynamics given a fixed topology.
The results are saved in the knstruc/ folder.

Topology reconstruction under known dynamical forms
Run topo_reconstruct_kndyn.py with different datasets to reconstruct network topology under fixed dynamics.
The results are stored in the kndyn/ folder.

Simultaneous inference
Run simultaneous_infer.py, setting m as the number of sampled time points.
Analyze the results using analyze.ipynb in the simul_infer/ folder.

