import os

import matplotlib.pyplot as plt
import numpy as np

from utils.file_utils import get_plots_dir

import matplotlib.pyplot as plt
import numpy as np

def plot_multi_latent_trajectory_pupil_v5(behavior_df, outputs, latent_indicies, workflow_name, num_stims=15, show_traj=True, save_plot=False, save_dir=None, show_plot=True, variances=None):
    """
    Plot latent trajectories from several trials on a single plot.

    Args:
    - behavior_df (pandas.DataFrame): DataFrame containing behavior data including 'frame' and 'pupil_size'.
    - outputs (numpy.ndarray): Latent data.
    - latent_indicies (numpy.ndarray): User provided latent indices.
    - workflow_name (str): Name of the workflow.
    - num_stims (int): Number of random chosen trials to plot.
    - save_plot (bool, optional): Whether to save the plot. Default is False.
    - save_dir (str, optional): Directory to save the plot. Required if save_plot is True.
    - show_plot (bool, optional): Whether to show the plot. Default is True.
    - variances (list, optional): List of variances for each latent dimension.

    Returns:
    - plots scatter where pupil size has color scale Greys in range [0,1]
    - plots trajectories of points for each trial plotted. trajectory labeled by frame with color scale Plasma in range [0s, 30s]
    """
    # Set plot font size
    fsz = 12

    # Extract pupil size data and normalize
    pupil_sizes = behavior_df['pupil_size']
    pupil_sizes_normalized = (pupil_sizes - pupil_sizes.min()) / (pupil_sizes.max() - pupil_sizes.min())

    # Assuming 'outputs' is your data
    latent_1_idx, latent_2_idx = latent_indicies
    print(f'Latent 1 idx: {latent_1_idx}')
    print(f'Latent 2 idx: {latent_2_idx}')

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Iterate over num_stims random stim_numbers
    stim_number_pool = np.random.choice(range(60), num_stims, replace=False)
    for i in range(num_stims):
        stim_number = stim_number_pool[i]
        latent_data_frequency = 5

        # Define point sizes based on pupil size with exponential increase for higher values
        max_point_size = 100
        min_point_size = 10
        point_sizes = min_point_size + (max_point_size - min_point_size) * pupil_sizes_normalized[900*stim_number:900*(stim_number+1)][::latent_data_frequency]

        # Plot the scatter plot with varying point size and hue based on pupil size
        ax.scatter(outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   s=point_sizes,
                   c=pupil_sizes_normalized[900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   cmap='Greys', alpha=0.7)  # Adjust alpha value if needed

        # Plot trajectory line with gradient color
        if show_traj:
            sample_frequency = 50
            x = outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
            y = outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
            time_indices = np.arange(len(x)) / len(x)  # Normalize time indices to [0, 1]
            colors = plt.cm.plasma(time_indices)  # Use colormap to create the gradient
            for j in range(len(x) - 1):  # Iterate over points to draw line segments with varying colors
                ax.plot(x[j:j+2], y[j:j+2], color=colors[j], linestyle='-', linewidth=1, alpha=0.5)

    # Set labels
    var_1 = "{:.3f}".format(variances[latent_1_idx]) if variances is not None else ''
    var_2 = "{:.3f}".format(variances[latent_2_idx]) if variances is not None else ''
    ax.set_xlabel(f'Latent {latent_1_idx} (var={var_1})', fontsize=fsz)
    ax.set_ylabel(f'Latent {latent_2_idx} (var={var_2})', fontsize=fsz)

    # Title
    if num_stims == 1:
        title = f"{workflow_name} - Trial #{stim_number_pool[0]}"
        ax.set_title(title, fontsize=fsz)
    else:
        ax.set_title(workflow_name, fontsize=fsz)

    if show_traj:
        # Add color bar for trajectory time
        cbar_trajectory = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax, orientation='vertical', shrink=0.7, pad=0.05, alpha=0.5)
        cbar_trajectory.set_label('Frame', fontsize=10, labelpad=0.1)
                
        # Customize color bar ticks and labels
        cbar_trajectory.set_ticks([0, 1])
        cbar_trajectory.set_ticklabels([f'0', f'900'])
        cbar_trajectory.ax.tick_params(labelsize=fsz)

    # Add color bar for pupil size
    cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='Greys'), ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Pupil Size', fontsize=10, labelpad=5)
            
    # Customize color bar ticks and labels
    cbar.ax.tick_params(labelsize=fsz)

    # Save or show the plot
    if save_plot:
        if save_dir:
            plt.savefig(f"{save_dir}/{workflow_name}.png")
        else:
            raise ValueError("Save directory not specified.")
    if show_plot:
        plt.show()

    plt.clf()

def plot_multi_latent_trajectory_pupil_v4(behavior_df, outputs, latent_indicies, workflow_name, num_stims=15, show_traj=True, save_plot=False, save_dir=None, show_plot=True, variances=None):
    """
    Plot latent trajectories from several trials on a single plot.

    Args:
    - behavior_df (pandas.DataFrame): DataFrame containing behavior data including 'frame' and 'pupil_size'.
    - outputs (numpy.ndarray): Latent data.
    - latent_indicies (numpy.ndarray): User provided latent indices.
    - workflow_name (str): Name of the workflow.
    - num_stims (int): Number of random chosen trials to plot.
    - save_plot (bool, optional): Whether to save the plot. Default is False.
    - save_dir (str, optional): Directory to save the plot. Required if save_plot is True.
    - show_plot (bool, optional): Whether to show the plot. Default is True.
    - variances (list, optional): List of variances for each latent dimension.

    Returns:
    - plots scatter where pupil size has color scale Greys in range [0,1]
    - plots trajectories of points for each trial plotted. trajectory labeled by frame with color scale Plasma in range [0s, 30s]
    """
    # Set plot font size
    fsz = 12

    # Extract pupil size data and normalize
    pupil_sizes = behavior_df['pupil_size']
    pupil_sizes_normalized = (pupil_sizes - pupil_sizes.min()) / (pupil_sizes.max() - pupil_sizes.min())

    # Assuming 'outputs' is your data
    latent_1_idx, latent_2_idx = latent_indicies
    print(f'Latent 1 idx: {latent_1_idx}')
    print(f'Latent 2 idx: {latent_2_idx}')

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Iterate over num_stims random stim_numbers
    stim_number_pool = np.random.choice(range(60), num_stims, replace=False)
    for i in range(num_stims):
        stim_number = stim_number_pool[i]
        latent_data_frequency = 5

        # Define point sizes based on pupil size with exponential increase for higher values
        max_point_size = 100
        min_point_size = 10
        point_sizes = min_point_size + (max_point_size - min_point_size) * pupil_sizes_normalized[900*stim_number:900*(stim_number+1)][::latent_data_frequency]

        # Plot the scatter plot with varying point size and hue based on pupil size
        ax.scatter(outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   s=point_sizes,
                   c=pupil_sizes_normalized[900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                   cmap='Greys', alpha=0.7)  # Adjust alpha value if needed

        # Plot trajectory line with gradient color
        if show_traj:
            sample_frequency = 50
            x = outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
            y = outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
            time_indices = np.arange(len(x)) / len(x)  # Normalize time indices to [0, 1]
            colors = plt.cm.plasma(time_indices)  # Use colormap to create the gradient
            for j in range(len(x) - 1):  # Iterate over points to draw line segments with varying colors
                ax.plot(x[j:j+2], y[j:j+2], color=colors[j], linestyle='-', linewidth=1, alpha=0.5)

    # Set labels
    var_1 = "{:.3f}".format(variances[latent_1_idx]) if variances is not None else ''
    var_2 = "{:.3f}".format(variances[latent_2_idx]) if variances is not None else ''
    ax.set_xlabel(f'Latent 1 (L_idx = {latent_1_idx}, var={var_1})', fontsize=fsz)
    ax.set_ylabel(f'Latent 2 (L_idx = {latent_2_idx}, var={var_2})', fontsize=fsz)

    # Title
    if num_stims == 1:
        title = f"{workflow_name} - Trial #{stim_number_pool[0]}"
        ax.set_title(title, fontsize=fsz)
    else:
        ax.set_title(workflow_name, fontsize=fsz)

    # Add color bar for trajectory time
    cbar_trajectory = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax, orientation='vertical', shrink=0.7, pad=0.05, alpha=0.5)
    cbar_trajectory.set_label('Trajectory Time', fontsize=10, labelpad=0.1)
            
    # Customize color bar ticks and labels
    cbar_trajectory.set_ticks([0, 1])
    cbar_trajectory.set_ticklabels([f'0 s', f'30 s'])
    cbar_trajectory.ax.tick_params(labelsize=fsz)

    # Add color bar for pupil size
    cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='Greys'), ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Pupil Size', fontsize=10, labelpad=5)
            
    # Customize color bar ticks and labels
    cbar.ax.tick_params(labelsize=fsz)

    # Save or show the plot
    if save_plot:
        if save_dir:
            plt.savefig(f"{save_dir}/{workflow_name}.png")
        else:
            raise ValueError("Save directory not specified.")
    if show_plot:
        plt.show()

    plt.clf()


def plot_multi_latent_trajectory(outputs, sorted_indices, workflow_name, num_stims=15, save_plot=False, save_dir=None, show_plot=True, variances=None):
    """
    Plot latent trajectories from several trials on a single plot.

    Args:
    - outputs (numpy.ndarray): Latent data.
    - sorted_indices (numpy.ndarray): Sorted indices.
    - workflow_name (str): Name of the workflow.
    - num_stims (int): Number of random chosen trials to plot.
    - save_plot (bool, optional): Whether to save the plot. Default is False.
    - save_dir (str, optional): Directory to save the plot. Required if save_plot is True.
    - show_plot (bool, optional): Whether to show the plot. Default is True.
    - variances (list, optional): List of variances for each latent dimension.

    Returns:
    - None
    """
    # Set plot font size
    fsz = 12

    # Assuming 'outputs' is your data
    latent_1_idx, latent_2_idx = sorted_indices[-2:]
    print(f'Latent 1 idx: {latent_1_idx}')
    print(f'Latent 2 idx: {latent_2_idx}')

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Iterate over num_stims random stim_numbers
    stim_number_pool = np.random.choice(range(60), num_stims, replace=False)
    for i in range(num_stims):
        stim_number = stim_number_pool[i]
        latent_data_frequency = 5

        # Plot the scatter plot with colorbar
        scatter = ax.scatter(outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                             outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::latent_data_frequency],
                             c=range(900)[::latent_data_frequency], cmap='coolwarm', s=10, alpha=0.7)  # Adjust marker size and transparency

        # Plot trajectory line with gradient color
        sample_frequency = 50
        x = outputs[0][:, latent_1_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
        y = outputs[0][:, latent_2_idx][900*stim_number:900*(stim_number+1)][::sample_frequency]  # Every 5th data point
        time_indices = np.arange(len(x)) / len(x)  # Normalize time indices to [0, 1]
        colors = plt.cm.plasma(time_indices)  # Use a reversed grayscale colormap to create the gradient
        for j in range(len(x) - 1):  # Iterate over points to draw line segments with varying colors
            ax.plot(x[j:j+2], y[j:j+2], color=colors[j], linestyle='-', linewidth=1)

    # Set labels
    var_1 = "{:.3f}".format(variances[latent_1_idx]) if variances is not None else ''
    var_2 = "{:.3f}".format(variances[latent_2_idx]) if variances is not None else ''
    ax.set_xlabel(f'Latent 1 (L_idx = {latent_1_idx}, var={var_1})', fontsize=fsz)
    ax.set_ylabel(f'Latent 2 (L_idx = {latent_2_idx}, var={var_2})', fontsize=fsz)

    # Title
    if num_stims == 1:
        title = f"{workflow_name} - Trial #{stim_number_pool[0]}"
        ax.set_title(title, fontsize=fsz)
    else:
        ax.set_title(workflow_name, fontsize=fsz)

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Frame', fontsize=10, labelpad=0.1)
            
    # Customize color bar ticks and labels
    cbar.set_ticks([0, 895])
    cbar.set_ticklabels([f'0 s', f'30 s'])
    cbar.ax.tick_params(labelsize=fsz)

    # Save or show the plot
    if save_plot:
        if save_dir:
            plt.savefig(f"{save_dir}/{workflow_name}.png")
        else:
            raise ValueError("Save directory not specified.")
    if show_plot:
        plt.show()

def plot_single_latent_trajectory(outputs, sorted_indices, workflow_name, save_plot=False, save_dir=None, show_plot=True):
    """
    Plot a single latent trajectory.

    Args:
    - outputs (numpy.ndarray): Latent data outputs.
    - sorted_indices (numpy.ndarray): Indices of sorted data.
    - workflow_name (str): Name of the workflow.
    - save_plot (bool, optional): Whether to save the plot. Default is False.
    - save_dir (str, optional): Directory to save the plot. Required if save_plot is True.
    - show_plot (bool, optional): Whether to display the plot. Default is True.
    """
    plot_multi_latent_trajectory(outputs, sorted_indices, workflow_name, num_stims=1, save_plot=save_plot, save_dir=save_dir, show_plot=show_plot)


def plot_validation_loss(s_n, workflow_name, version_number, save_plot=False, show_plot=True):
    """
    Plot validation loss.

    Args:
    - s_n: History of validation loss.
    - workflow_name (str): Name of the workflow.
    - version_number (int): Version number.
    - save_plot (bool, optional): Whether to save the plot. Default is False.
    - show_plot (bool, optional): Whether to display the plot. Default is True.
    """
    plot_dir = get_plot_dir(workflow_name, version_number, plot_type='metric')

    fig, ax = plt.subplots(figsize=(6, 4))  # Increase figure size
    fsz = 12

    ax.plot(s_n.history['val_loss'][:])

    ax.set_xlabel('Epochs', fontsize=fsz)
    ax.set_ylabel('Validation Loss', fontsize=fsz)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.patch.set_facecolor('white')  # Set background color for the figure

    plt.setp(ax.get_xticklabels(), fontsize=fsz)  # Remove background color for the tick labels
    plt.setp(ax.get_yticklabels(), fontsize=fsz)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusted tight layout with more space for title

    plt.title(f"{workflow_name} - Val Loss", fontsize=fsz)

    if save_plot:
        plt.savefig(f"{plot_dir}/{workflow_name}_val_loss.png", dpi=300)
    if not show_plot:
        plt.close()
    elif show_plot:
        plt.show()

def plot_reconstruction_errors(reconstruction_errors, brain_region):
    """
    Plot reconstruction errors for workflows.

    Args:
    - reconstruction_errors (dict): A dictionary where keys are workflow names and values are reconstruction errors.
    - brain_region (str): The name of the brain region being analyzed.

    Returns:
    - None
    """

    # Sort the workflows by reconstruction error
    sorted_workflows = sorted(reconstruction_errors.items(), key=lambda x: x[1])

    # Extract workflow names and reconstruction errors for plotting
    workflows = [wf[0] for wf in sorted_workflows]
    errors = [wf[1] for wf in sorted_workflows]

    # Create title based on brain region
    title = f"Comparison of Reconstruction Errors Across {brain_region} Brain Region Workflows"

    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(workflows, errors, color='skyblue')
    plt.xlabel('Reconstruction Error')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to display highest error at the top
    plt.show()



def get_plot_dir(workflow_name, version_number, plot_type='latent'):
    """
    Generate the directory path for storing plots based on model parameters and version number.

    Args:
    - workflow_name (str): Type of model workflow.
    - version_number (int): Version number or iteration number.
    - plot_type (str, optional): Type of plot. Default is 'latent'.

    Returns:
    - plot_dir (str): Path for storing plots.
    """
    print('hi from plotting')
    
    # Define the base directory for plots
    base_dir = get_plots_dir()
    
    # Construct the directory path based on model parameters
    model_dir_name = workflow_name
    model_dir = os.path.join(base_dir, model_dir_name)
    
    # Create subdirectory for the specified version number
    version_dir = os.path.join(model_dir, f"v{version_number}")
    
    # Create subdirectory for the specified plot type
    plot_dir = os.path.join(version_dir, plot_type)
    
    # Create directories if they don't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    return plot_dir