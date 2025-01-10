# Dissipation CNN

Convolutional Neural Network to extract dissipation rate from VMP-250 dataset

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Output](#output)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Examples](#examples)

## Introduction

The Dissipation CNN prediction project is designed to streamline the process in processing data from the VMP-250. 
The project takes in the raw .MAT data from the VMP, and will output the dissipation rate calculations, and shear spectra.
This repository provides tools for analyzing dissipation rates of ocean shear data, integrating MATLAB-derived methods into a Python workflow. It includes:

- Data loading and processing scripts
- A machine learning model (trained with Keras/TensorFlow) to predict integration ranges for computing dissipation rates
- Interactive visualization tools for exploring shear spectra and adjusting integration ranges

The CNN is designed to identify the optimal integration range in order to compute the dissipation rate over specific windows in the captured data. 
After it has computed the integration range, it uses the following formula:

$$\epsilon_{\text{cnn}} = 7.5 \cdot \nu \cdot \int_{K_{\min}}^{K_{\max}} P_{\text{sh}}(K) \, dK$$

The model also outputs a confidence interval as to the quality of the data, which is represented from 0 - 1 in the `flagood` output variable.

## Features

- **Input**:  
  Load `.mat` files containing `data` and `dataset` variables, extract relevant profile information, and process ocean shear data.

- **Signal Processing**:  
  Apply filtering, to remove acceleration-coherent noise from shear signals.

- **Machine Learning Integration**:  
  Train or load a CNN model to predict the optimal integration range in order to calculate the dissipation rate in the window.

- **Interactive Plotting**:  
  Use a matplotlib-based interactive interface to visualize shear spectra, adjust integration, and see the recomputed dissipation results.

## Output
 Returns `.mat` file which includes:
 - Dissipation `e`
 - Integration Range Wavnumbers `K_min` and `K_max`
 - Nasmyth Spectra `Nasmyth_spec`
 - Shear `Sh`
 - Cleaned shear spectra`sh_clean`
 - Acceleration Signals`A`
 - Cross-spectra between shear and acceleration `UA`
 - Frequency Vector `F`
 - Wavenumber Array `K`
 - Profiling speed `speed`
 - Kinematic Viscosity`nu`
 - Pressure `P`
 - Temperature `T`
 - Buoyancy Frequency `N2`
 - Eddy diffusivity of density`Krho`
 - Data Flag `flagood`


## Requirements

- Python 3.8+
- matplotlib
- numpy
- scipy
- h5py
- mat73
- gsw
- tensorflow
- scikit-learn
- keras

*See `requirements.txt`*

## Installation

Step-by-step instructions to install dependencies and set up the environment:

`git clone https://github.com/taikil/ML-Lab.git`

`cd ML-Lab`



`pip install -r requirements.txt`

# Usage

1. Data Preparation:
    Ensure you have .mat files containing data and dataset variables. These should be MATLAB v7.3 format or convertible using mat73.

2. Model Selection:
   - If you have a pre-trained model, provide its name when prompted.
   - If not, choose to train a new model or continue training an existing model.

3. Running the Script:
   Run the main script:

`python3 predict_dissipation.py path_to_your_mat_file.mat`

Follow the prompts:

- Enter the model filename (without extension).
- Choose whether to continue training, train new, or use an existing model without retraining.
- Enter the profile number to process.

4. Interactive Spectra Viewing:
After processing, an interactive matplotlib window appears.

- Use Next and Previous buttons to navigate through different spectra.
- Click Reselect Range to manually pick new integration ranges:
  - First click sets start of integration range.
  - Second click sets end of integration range.

The dissipation rate is then recomputed with the updated range.

# Model Training
The training of the model is handled with the preprocessed data files, such as those already outputted from the existing matlab code. The machine takes the following two inputs:

`spectrum_input = [P_sh_log, P_nasmyth_log, K_normalized]`
`scalar_features = [nu, P, T]`

Where `P_sh_log` is the log normalized shear spectrum, `P_namysmth_log` is the log normalized Nasmyth spectrum, and `K_normalized` is the normalized wavenumber array.

The scalar features consists of the kinematic viscosity, the pressure and temperature

The format of the CNN is as follows:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ spectrum_input (InputLayer)   │ (None, 1025, 3)           │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d (Conv1D)               │ (None, 1025, 64)          │           1,024 │ spectrum_input[0][0]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization           │ (None, 1025, 64)          │             256 │ conv1d[0][0]               │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ max_pooling1d (MaxPooling1D)  │ (None, 512, 64)           │               0 │ batch_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_1 (Conv1D)             │ (None, 512, 128)          │          41,088 │ max_pooling1d[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_1         │ (None, 512, 128)          │             512 │ conv1d_1[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ max_pooling1d_1               │ (None, 256, 128)          │               0 │ batch_normalization_1[0][… │
│ (MaxPooling1D)                │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ conv1d_2 (Conv1D)             │ (None, 256, 256)          │          98,560 │ max_pooling1d_1[0][0]      │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ batch_normalization_2         │ (None, 256, 256)          │           1,024 │ conv1d_2[0][0]             │
│ (BatchNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ global_average_pooling1d      │ (None, 256)               │               0 │ batch_normalization_2[0][… │
│ (GlobalAveragePooling1D)      │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ scalar_input (InputLayer)     │ (None, 3)                 │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ concatenate (Concatenate)     │ (None, 259)               │               0 │ global_average_pooling1d[… │
│                               │                           │                 │ scalar_input[0][0]         │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 128)               │          33,280 │ concatenate[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 64)                │           8,256 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ flagood_output (Dense)        │ (None, 1)                 │              65 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ integration_output (Dense)    │ (None, 2)                 │             130 │ dense_1[0][0]              │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
```

