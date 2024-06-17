# src/modeling/vae_model.py
import sys
src = "/Users/rp/Desktop/Research/CN^3/Thesis Material/2-Region-Latent-Alignment/src"
if src not in sys.path:
    sys.path.append(src)

import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from pi_vae.pi_vae import *  # Assuming vae_mdl is a function in pi_vae module
from pi_vae.util import *
from utils.file_utils import get_models_dir

# Import Logging
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def initialize_model(dim_x, dim_z, dim_u, gen_nodes, n_blk, mdl, disc, learning_rate, random_seed=None):
    """
    Initialize the Variational Autoencoder (VAE) model.

    Args:
    - dim_x (int): Dimensionality of input data.
    - dim_z (int): Dimensionality of latent space.
    - dim_u (int): Dimensionality of metadata.
    - gen_nodes (int): Number of nodes in the generator layers.
    - n_blk (int): Number of blocks in the model.
    - mdl (str): Type of model (e.g., 'poisson').
    - disc (bool): Whether the model is discriminative.
    - learning_rate (float): Learning rate for optimization.
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - vae (keras.Model): Initialized VAE model.
    """
    if random_seed is None:
        raise ValueError("Random seed must be specified.")
    
    np.random.seed(random_seed)
    vae = vae_mdl(dim_x=dim_x, dim_z=dim_z, dim_u=dim_u, gen_nodes=gen_nodes, n_blk=n_blk, mdl=mdl, disc=disc, learning_rate=learning_rate)
    return vae


def train_model(vae, x_train, u_train, x_valid, u_valid, model_chk_path):
    """
    Train the VAE model.

    Args:
    - vae (keras.Model): Initialized VAE model.
    - x_train (numpy.ndarray): Training data.
    - u_train (numpy.ndarray): Training metadata.
    - x_valid (numpy.ndarray): Validation data.
    - u_valid (numpy.ndarray): Validation metadata.
    - model_chk_path (str): File path to save model checkpoints.

    Returns:
    - s_n (object): Training history object.
    """
    mcp = ModelCheckpoint(model_chk_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
    s_n = vae.fit_generator(custom_data_generator(x_train, u_train),
                            steps_per_epoch=len(x_train), epochs=50, 
                            verbose=1,
                            validation_data=custom_data_generator(x_valid, u_valid),
                            validation_steps=len(x_valid), 
                            callbacks=[mcp])
    return s_n

def inference(vae, x_all, u_all, model_chk_path):
    """
    Perform inference using the trained VAE model.

    Args:
    - vae (keras.Model): Trained VAE model.
    - x_all (numpy.ndarray): Input data for inference.
    - u_all (numpy.ndarray): Metadata for inference.
    - model_chk_path (str): Path to the model checkpoint file.

    Returns:
    - outputs (numpy.ndarray): Inference outputs.
    """
    # Load the weights
    vae.load_weights(model_chk_path)
    
    # Perform inference
    outputs = vae.predict_generator(custom_data_generator(x_all, u_all), steps=len(x_all))
    
    return outputs

# Additional functions for monitoring and plotting loss during training can be added here

def get_model_paths(session_id, brain_region, selected_behavior_vars, latent_dim, random_seed, model_type='pivae'):
    """
    Generate paths for model checkpoint and workflow name.

    Args:
    - brain_region (str): Brain region for which the model is trained.
    - selected_behavior_vars (str or list of str): Selected behavior variables.
    - latent_dim (int): Dimensionality of the latent space.
    - random_seed (int): Random seed used for model training.
    - model_type (str, optional): Type of model workflow. Default is 'pivae'.

    Returns:
    - model_chk_path (str): Path for model checkpoint file.
    - workflow_name (str): Name of the workflow.
    """
    model_dir = get_models_dir()
    model_subdir = f"{model_type}_{latent_dim}d_{random_seed}"
    
    # Create the directory if it doesn't exist
    model_dir = os.path.join(model_dir, f"session_{session_id}", model_subdir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Join selected behavior variables if it's a list
    if isinstance(selected_behavior_vars, list):
        selected_behavior_vars = '_'.join(selected_behavior_vars)

    # Create subdirectory based on brain region
    model_dir = os.path.join(model_dir, brain_region)
    os.makedirs(model_dir, exist_ok=True)
    
    workflow_name = f"{model_type}_{latent_dim}d_{random_seed}_{brain_region}_{selected_behavior_vars}"
    model_chk_path = os.path.join(model_dir, f"{workflow_name}.h5")
    
    return model_chk_path, workflow_name