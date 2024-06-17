# Pi-VAE-for-Visual-Stimuli
Applying Pi-VAE to the Neuropixels Visual Coding Dataset.

Repository for Master's thesis: "Interpreting Latent Manifolds of High-Dimensional Neural Activity Using Pi-VAE"

This repository contains the research project focused on constructing latent manifolds of high-dimensional neuron spike activity, aiming to generate low-dimensional latent plots displaying geometric structure indicative of salient factors capturing neuron behavior. The directory structure is organized to facilitate clarity and accessibility of various project components.


## Running Code
The code for the models is written in Python 3.6. In addition to standard scientific Python libraries (numpy, scipy, matplotlib), the code expects the tensorflow (1.13.1) and keras (2.3.1) packages.
(This is directly taken from code source.)

  #### Note
  Many notebooks were run with the packages provided by requirements.txt file. However to get replicable model results as source paper (for this TF/Keras implementation) I recommend using requirements from source as a baseline.
  - TODO: Distill requirements.txt

## Model
Pi-VAE model implementation is directly inherited from original paper

Zhou, D., Wei, X. Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE. NeurIPS 2020. https://arxiv.org/abs/2011.04798

Source code found here: https://github.com/zhd96/pi-vae

## Data source
Allen Institute Visual Coding – Neuropixels Project
  - https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels
  - https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels

  SDK Documentation page: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html

NOTE: If using processed `.h5` files, the key parameter to open is 'pivot' or 'key'. If this doesn't work please use `.csv` files, data is the same.

## Directory Structure

  - **`data/`:**
    - Contains raw and processed data files utilized in the research.
      - `raw/`: Raw data files before preprocessing.
      - `processed/`: Processed data files ready for analysis.

  - **`models/`:**
    - Stores trained models organized by hyperparameter combinations.
      - `pivae_2d_777/`, `pivae_2d_999/`, `pivae_10d_777/`, `pivae_10d_999/`: Trained models using PIVAE architecture with different latent dimension sizes (2D, 10D) and random seeds.
        - Subdirectories for each brain region/behavior variable combination.
      - Similar directories exist for other model types (e.g., VAE) and hyperparameter combinations.
  - **`src/`:**
    - Houses modularized and packaged code for the modeling chunk.
      - `modeling/`: Python package for model initialization, training, and evaluation.
        - `__init__.py`: Initialization file for the package.
        - `model.py`: Code for initializing and training the model.
        - `utils.py`: Utility functions specific to the modeling process.

      - `notebooks/`: Contains Jupyter notebooks used for data exploration, model training, and visualization.
          - `exploratory/`: Notebooks for initial data analysis and exploration.
          - `modeling/`: Notebooks for model training and tuning.
          - `preprocessing/`: Notebooks for ingesting and preprocessing data.

      - `pi_vae/`: Contains implementation of Pi-VAE and VAE models directly taken from Zhou, D., Wei, X. code here: https://github.com/zhd96/pi-vae

      - `utils/`: Contains utility Python files for various project tasks.
          - `plotting.py`: Functions for saving plots.
          - `data_processing.py`: Functions for data preprocessing.
          - `model_utils.py`: Utility functions for model training and evaluation.
          - `file_utils.py`: Utility functions for required file paths.
          - `helper.py`: Broad set of support functions, from data preprocessing to visualization.

  This structured directory layout aims to enhance project organization, making it easier to navigate, understand, and contribute to the research project on aligning latent representations of neural activity across different brain regions and behavioral variables.
