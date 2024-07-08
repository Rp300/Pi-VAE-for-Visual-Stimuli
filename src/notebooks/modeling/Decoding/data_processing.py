import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from utils.file_utils import get_data_dir
from keras import backend as K


FRAMES_PER_MOVIE = 900


def compute_marginal_lik_single_batch(vae_mdl, y_test_batch, u_fake, n_sample, pbar):
    lik_test = []
    for ii in range(u_fake.shape[0]):  # For each unique u value
        # Properly selecting and reshaping the u value for prediction
        u_value = u_fake[ii].reshape(-1, 1)  # Assuming the VAE expects a shape like (n_samples, n_features)
        opts = vae_mdl.predict([y_test_batch, u_value])
        lam_mean = opts[4]
        lam_log_var = opts[5]
        z_dim = lam_mean.shape
        z_sample = np.random.normal(0, 1, size=(n_sample, z_dim[0], z_dim[1]))
        z_sample = z_sample * np.exp(0.5 * lam_log_var) + lam_mean

        # Compute fire rate
        get_fire_rate_output = K.function([vae_mdl.layers[-1].get_input_at(0)],
                                          [vae_mdl.layers[-1].get_output_at(0)])
        fire_rate = get_fire_rate_output([z_sample.reshape(-1, z_dim[-1])])[0]
        fire_rate = fire_rate.reshape(n_sample, -1, fire_rate.shape[-1])

        # Compute p(x|z) Poisson likelihood
        loglik = y_test_batch * np.log(np.clip(fire_rate, 1e-10, 1e7)) - fire_rate
        loglik = loglik.sum(axis=-1)  # sum across neurons
        loglik_max = loglik.max(axis=0)
        loglik -= loglik_max
        tmp = np.log(np.exp(loglik).mean(axis=0)) + loglik_max
        lik_test.append(tmp)

        # Update progress bar after each u value processed
        # if pbar:
        pbar.update(1)
    
    return np.array(lik_test)

# def extract_data(ecephys_session_id, brain_region):
#     """
#     Extract spike count and behavior data for a given session and brain region.

#     Args:
#     - ecephys_session_id (int): The ecephys Functional Connectivity session ID.
#     - brain_region (str): The brain region to extract data for (e.g., "VISp", "CA1").

#     Returns:
#     - spike_count_pivot (pd.DataFrame): Spike count data pivoted by neurons and time bins.
#     - behavior_data_df (pd.DataFrame): DataFrame containing behavior data.
#     """
#     # Get the data directory path
#     data_path = get_data_dir()

#     # Construct session directory path
#     session_path = f"session_{ecephys_session_id}/"
#     processed_path = "processed/"
#     directory_path = os.path.join(data_path, processed_path, session_path)

#     # Check if the directory exists
#     if not os.path.exists(directory_path):
#         raise FileNotFoundError(f"Directory {directory_path} does not exist")

#     # Read spike count data pivoted by neurons and time bins
#     spike_data_file = f'{brain_region}_spike_count_{ecephys_session_id}_pivot.h5'
#     spike_file_path = os.path.join(directory_path, spike_data_file)
#     if not os.path.exists(spike_file_path):
#         raise FileNotFoundError(f"File {spike_file_path} does not exist")
#     spike_count_pivot = pd.read_hdf(spike_file_path, key='df')

#     # Read behavior data
#     behavior_data_file = f'NaturalMovie_Behavior_{ecephys_session_id}_normalized.h5'
#     behavior_file_path = os.path.join(directory_path, behavior_data_file)
#     if not os.path.exists(behavior_file_path):
#         raise FileNotFoundError(f"File {behavior_file_path} does not exist")
#     behavior_data_df = pd.read_hdf(behavior_file_path, key='df')

#     return spike_count_pivot, behavior_data_df


def transform_data(spike_count_pivot, behavior_data_df, selected_behavior_vars):
    """
    Transform spike count and behavior data into input arrays for modeling.

    Args:
    - spike_count_pivot (pd.DataFrame): Spike count data pivoted by neurons and time bins.
    - behavior_data_df (pd.DataFrame): DataFrame containing behavior data.
    - selected_behavior_vars (list): List of selected behavior variables.

    Returns:
    - x_all (numpy.ndarray): Array containing neuron spike activity data.
    - u_all (numpy.ndarray): Array containing labels or metadata.
    """
    global FRAMES_PER_MOVIE  # Assuming FRAMES_PER_MOVIE is a global variable

    # Reshape spike count data
    x_all = spike_count_pivot.values.reshape(-1, FRAMES_PER_MOVIE, spike_count_pivot.shape[1])

    # Transform behavior data
    u_all = behavior_data_df[selected_behavior_vars].to_numpy().reshape(-1, FRAMES_PER_MOVIE, len(selected_behavior_vars))

    return x_all, u_all


def load_data(x_all, u_all, train_ratio=0.8, valid_ratio=0.1):
    """
    Load data from input variables x_all and u_all using train_test_split.

    Args:
    - x_all (numpy.ndarray): Array containing neuron spike activity data.
    - u_all (numpy.ndarray): Array containing labels or metadata.
    - train_ratio (float, optional): Ratio of data to be allocated for training.
                                     Default is 0.8 (80%).
    - valid_ratio (float, optional): Ratio of data to be allocated for validation from the remaining, non-training, data.
                                     Default is 0.1 (10%).

    Returns:
    - x_train (numpy.ndarray): Training data.
    - u_train (numpy.ndarray): Labels or metadata for training data.
    - x_valid (numpy.ndarray): Validation data.
    - u_valid (numpy.ndarray): Labels or metadata for validation data.
    - x_test (numpy.ndarray): Testing data.
    - u_test (numpy.ndarray): Labels or metadata for testing data.
    """

    # Calculate the size of the training set
    train_size = int(train_ratio * len(x_all))

    # Split the data into training and remaining sets
    X_train, X_remain, u_train, u_remain = train_test_split(x_all, u_all, train_size=train_size)

    # Calculate the sizes of the validation and test sets
    valid_size = int(valid_ratio * len(x_all))
    test_size = len(x_all) - train_size - valid_size

    # Split the remaining data into validation and test sets
    X_valid, X_test, u_valid, u_test = train_test_split(X_remain, u_remain, test_size=test_size)

    return X_train, u_train, X_valid, u_valid, X_test, u_test