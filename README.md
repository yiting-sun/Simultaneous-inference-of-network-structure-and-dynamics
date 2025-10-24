# GENSIE - Governing Equation and Network Simultaneous Inference Engine
Codes and data for paper: Simultaneous Inference of Network Structure and Dynamics:How Much Data is Needed?

Authors: Yi-Ting Sun, Ting-Ting Gao*, Gang Yan*

## Description

We introduce a unified framework that achieves simultaneous inference of network topology and governing dynamical equations from time-series observations, combining interpretable mathematical expressions with computational flexibility. Through rigorous analysis of coupled oscillator networks, we discover fundamental limits of the amount of required data for accurate inference and identify the phase transition from non-inferrability to inferrability across network sizes. Our results demonstrate distinct regimes where either dynamical learning or structural reconstruction dominates data needs. This work provides both theoretical foundations and practical guidelines for data-driven network inference across scientific domains.

## GENSIE

This repository shows GENSIE for simultaneously inferring network dynamics and structure, and fundamental data limits analysis further. For rapid usage of this framework, you can run 'Simultaneous_Inference_HR.ipynb' in the GENSIE4signHR folder. 
Before starting, the 'data_path' and 'save_path' are needed to correspond to your path.

## Requirements
This framework requires Python 3.10 or higher, as well as several common scientific computing libraries, as shown in environment.yml and requirements.txt. These libraries can be installed.
If you use **conda**:
`conda env create -f environment.yml`
`conda activate your_env_name`
Alternatively, if you are using **pip**:
`pip install -r requirements.txt`

## Contact
- For questions or issues, please contact: yiting_sun@tongji.edu.cn
- Or open an issue on this repository
