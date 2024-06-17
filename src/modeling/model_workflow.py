'''
The goal here is to have a single file that can be prompted with command line args. It should complete all 3 steps of the modeling workflow: Preprocessing, Training, and Inference.

- The relevant code already exists in pivae_VISp_frame.ipynb
- inputs need to be the Global vars of the notebook.
- begin by pasting all code
'''

## Imports ##
import sys
src = "/Users/rp/Desktop/Research/CN^3/Thesis Material/2-Region-Latent-Alignment/src"
if src not in sys.path:
    sys.path.append(src)

from utils.file_utils import add_project_dirs_to_path

# Add project root and src directory to Python path
add_project_dirs_to_path()

# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import from project modules
from modeling.vae_model import *
from utils.data_processing import *
from utils.plotting import *
from utils.model_utils import *

# Import Keras callback
from keras.callbacks import ModelCheckpoint

# Import plot packages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import seaborn

# Import Logging
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")
##############################################################################

# Global Vars
VERSION = 1
ECEPHYS_SESSION_ID = 766640955 
FRAMES_PER_MOVIE = 900

LATENT_DIM = 6
RANDOM_SEED = 999
BRAIN_REGIONS = ['CA1', 'VISp']
MODEL_TYPES = ['vae', 'pivae']
TASK_VARIABLES = ['frame', 'pupil_size', 'position', 'total_distance']

# Save Plots flags
SAVE_VAL_PLOTS = True
SAVE_LATENT_PLOTS = True
SAVE_CORR_PLOTS = True

# Show Plots flags
SHOW_VAL_PLOTS = False
SHOW_LATENT_PLOTS = False
SHOW_CORR_PLOTS = False

# Flag for if this is a training iteration
TRAINING = False

def validate_inputs(brain_region, selected_behavior_vars, b_vars_to_plot):
    # Check if brain_region is None or not in BRAIN_REGIONS
    if brain_region is None or brain_region not in BRAIN_REGIONS:
        raise ValueError("Enter a valid brain region from: {}".format(BRAIN_REGIONS))
    
    # Check if selected_behavior_vars and b_vars_to_plot are empty or None
    if not selected_behavior_vars or not b_vars_to_plot:
        raise ValueError("Selected behavior variables or plotting variables cannot be empty.")
    
    # Check if elements in selected_behavior_vars exist in TASK_VARIABLES
    for var in selected_behavior_vars:
        if var not in TASK_VARIABLES:
            raise ValueError("Selected behavior variable '{}' not found in TASK_VARIABLES.".format(var))
    
    # Check if elements in b_vars_to_plot exist in TASK_VARIABLES
    for var in b_vars_to_plot:
        if var not in TASK_VARIABLES:
            raise ValueError("Plotting variable '{}' not found in TASK_VARIABLES.".format(var))


def execute_workflow(brain_region, selected_behavior_vars, b_vars_to_plot, ecephys_session_id=ECEPHYS_SESSION_ID):
    ## Validate inputs ##
    logging.info("Validating inputs...")
    validate_inputs(brain_region, selected_behavior_vars, b_vars_to_plot)
    
    ## Begin Workflow ##
    logging.info("Beginning workflow...")
    
    # Extract data
    logging.info("Extracting data...")
    spike_count_pivot, behavior_data_df = extract_data(ecephys_session_id, brain_region)
    logging.info("Data extraction complete.")
    
    # Transform data
    logging.info("Transforming data...")
    x_all, u_all = transform_data(spike_count_pivot, behavior_data_df, selected_behavior_vars)
    logging.info("Data transformation complete.")
    
    ## Load Data ##
    logging.info("Loading data...")
    x_train, u_train, x_valid, u_valid, x_test, u_test = load_data(x_all=x_all, u_all=u_all)
    logging.info("Data loading complete.")
    
    ## Initialize Model ##
    logging.info("Initializing model...")
    # Get model path and workflow name
    model_chk_path, workflow_name = get_model_paths(brain_region=brain_region,
                                        selected_behavior_vars=selected_behavior_vars,
                                        latent_dim=LATENT_DIM, 
                                        random_seed=RANDOM_SEED, 
                                        model_type=MODEL_TYPES[1])
    # Initialize model
    vae = initialize_model(dim_x=x_all[0].shape[-1], 
                        dim_z=LATENT_DIM,
                        dim_u=u_all[0].shape[-1], 
                        gen_nodes=60, n_blk=2, mdl='poisson', disc=False, learning_rate=5e-4, random_seed=RANDOM_SEED)
    logging.info("Model initialized.")
    
    ## Training ##
    if TRAINING:
        logging.info("Training model...")
        s_n = train_model(vae, x_train, u_train, x_valid, u_valid, model_chk_path)
        logging.info("Training complete.")

        logging.info("Plotting validation loss...")
        plot_validation_loss(s_n, workflow_name, VERSION, save_plot=SAVE_VAL_PLOTS, show_plot=SHOW_VAL_PLOTS)

    # Results
    logging.info("Performing inference...")
    outputs = inference(vae, x_all, u_all, model_chk_path) # post_mean, post_log_var, z_sample,fire_rate, lam_mean, lam_log_var, z_mean, z_log_var
    logging.info("Inference complete.")

    ## Compute variances ##
    logging.info("Computing variances...")
    # Calculate variances of all columns of outputs[0]
    variances = outputs[0].var(axis=0)

    # Sort the columns by variance in descending order and get the indices
    sorted_indices = np.argsort(variances)[::-1]

    # Create a new object containing the columns of outputs[0] sorted by variance
    # sorted_outputs = outputs[0][:, sorted_indices]
    logging.info("Variances computed.")

    ## Plotting ##

    ## Plot Latents
    # Get latent plotting dir
    if SHOW_LATENT_PLOTS or SAVE_LATENT_PLOTS:
        logging.info("Plotting latents...")
        plot_dir_latent = get_plot_dir(workflow_name, version_number=VERSION, plot_type='latent')
        logging.info(f"latent plot directory: {plot_dir_latent}")

        latent_indices = sorted_indices[-2:] # Latents with smallest variance

        plot_multi_latent_trajectory_pupil_v4(behavior_data_df, outputs, latent_indices, workflow_name, variances=variances, num_stims=15, show_traj=True, save_plot=SAVE_LATENT_PLOTS, save_dir=plot_dir_latent, show_plot=SHOW_LATENT_PLOTS)
        logging.info("Latents plotted.")

    ## Plot latent correlation/metrics
    # Get metric plotting dir
    if SHOW_CORR_PLOTS or SAVE_CORR_PLOTS:
        logging.info("Plotting latent correlation/metrics...")
        plot_dir_metric = get_plot_dir(workflow_name, version_number=VERSION, plot_type='metric')

        num_trials_to_plot = 60
        start_trial_id = 0

        start_idx = FRAMES_PER_MOVIE*start_trial_id
        end_idx = FRAMES_PER_MOVIE*num_trials_to_plot+FRAMES_PER_MOVIE*start_trial_id

        latents = {f"Latent {latent_idx}": outputs[0][:, latent_idx][start_idx : end_idx] for latent_idx in range(LATENT_DIM)}

        statistical_proving_latent_behavior_correlation(behavior_data_df, 
                                                        b_vars_to_plot, latents,   
                                                        workflow_name, r_squared_threshold=0.1,show_plots=SHOW_CORR_PLOTS, save_plots=SAVE_CORR_PLOTS, save_path=plot_dir_metric)
        logging.info("Latent correlation/metrics plotted.")
    
    return workflow_name


def main():
    logging.info("Let's Begin!")
    # Define different combinations of brain regions and behavior/task variables
    workflows_VISp = [ 
        {
            'brain_region': 'VISp', 
            'behavior_vars': ['frame'], 
            'behavior_to_plot': ['frame', 'pupil_size']
        },
        {
            'brain_region': 'VISp', 
            'behavior_vars': ['pupil_size'], 
            'behavior_to_plot': ['frame', 'pupil_size']
        },
        {
            'brain_region': 'VISp', 
            'behavior_vars': ['frame', 'pupil_size'], 
            'behavior_to_plot': ['frame', 'pupil_size']
        }
    ]

    workflows_CA1 = [ 
        {
            'brain_region': 'CA1', 
            'behavior_vars': ['frame'], 
            'behavior_to_plot': ['frame', 'position']
        },
        {
            'brain_region': 'CA1', 
            'behavior_vars': ['position'], 
            'behavior_to_plot': ['frame', 'position']
        },
        {
            'brain_region': 'CA1', 
            'behavior_vars': ['total_distance'], 
            'behavior_to_plot': ['frame', 'position']
        }
    ]
    
    # Loop through workflows and execute them
    for workflow in workflows_CA1:
        # Extract data
        brain_region = workflow['brain_region']
        behavior_vars = workflow['behavior_vars']
        behavior_to_plot = workflow['behavior_to_plot']
        try:
            workflow_name = execute_workflow(brain_region=brain_region, 
                                             selected_behavior_vars=behavior_vars, b_vars_to_plot=behavior_to_plot, ecephys_session_id=ECEPHYS_SESSION_ID)
            
            print("Workflow executed successfully for brain region:", workflow_name)
        except ValueError as e:
            print("Error executing workflow for brain region '{}': {}".format(brain_region, e))


if __name__ == "__main__":
    main()