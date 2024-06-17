import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def convertMillis(millis):
    seconds=int(millis/1000)%60
    minutes=int(millis/(1000*60))%60
    hours=int(millis/(1000*60*60))%24
    return seconds, minutes, hours

def print_time(millis):
    con_sec, con_min, con_hour = convertMillis(int(millis))
    print(f"{con_hour:02d}:{con_min:02d}:{con_sec:02d}")

def user_prompted_time():
    millis=input("Enter time in milliseconds ")
    print_time(millis)

########################################################################    

def run_speed_for_stim(row, run_speed_df):
    stim_speed_df = run_speed_df[(run_speed_df['start_time'] <= row['stop_time']) & (run_speed_df['end_time'] >= row['start_time'])]
    avg_speed = np.nan if stim_speed_df.empty else np.mean(stim_speed_df['velocity'])
    return avg_speed

def chunk_movies(movies_df):
    chunk_size = 900
    num_rows = len(movies_df)

    # Calculate the number of chunks needed
    num_chunks = int(np.ceil(num_rows / chunk_size))

    # Create a list to store the chunks
    movie_chunks = [movies_df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    return movie_chunks

def compute_relative_positions(movie_chunks):
    chunk_positions = []
    for movie_chunk in movie_chunks:
        movie_chunk_positions = movie_chunk['velocity'].cumsum()
        adj_rat_positions = movie_chunk_positions - movie_chunk_positions.iloc[0]
        chunk_positions.append(adj_rat_positions)
    return pd.concat(chunk_positions, axis=0)

def compute_position(movies_df):
    return compute_relative_positions(chunk_movies(movies_df))

#############################################################################
## Plot Kinematics attributes:   Velocity, Position, Total Distance
#############################################################################

def plot_kinematics(behavior_data_df, stim_number, window_size=900):
    fsz=14

    window = window_size

    # Get data for the selected stim_number
    position = behavior_data_df[behavior_data_df['trial'] == stim_number]['position'][:window]
    velocity = behavior_data_df[behavior_data_df['trial'] == stim_number]['velocity'][:window]
    total_distance = behavior_data_df[behavior_data_df['trial'] == stim_number]['total_distance'][:window]

    # Create separate figures and axes for each variable
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # print(len(velocity), len(position), len(total_distance))

    # Plot velocity in red
    axs[0].plot(np.linspace(0,30,900)[:window], velocity, color='red')
    axs[0].set_ylabel('Velocity (cm/s)', fontsize=fsz)

    # Plot position in blue
    axs[1].plot(np.linspace(0,30,900)[:window], position, color='blue')
    axs[1].set_ylabel('Position', fontsize=fsz)

    # Plot total distance in green
    axs[2].plot(np.linspace(0,30,900)[:window], total_distance, color='green')
    axs[2].set_ylabel('Total Distance', fontsize=fsz)

    # Set common x-axis label
    for ax in axs:
        ax.set_xlabel('Frame timestamp (s)', fontsize=fsz)

    # Title for all plots
    title = f"Stimulus Number: {stim_number}"
    fig.suptitle(title, fontsize=fsz+2)

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()


#############################################################################
## Plot/Describe Attribute Statistics: mean, median, std_dev, variance, min, max, quartiles
#############################################################################

def analyze_single_trial(behavior_df, trial_number, attribute_name):
    """
    Analyze the 'total_distance' values for a specific trial number from a DataFrame.
    
    Parameters:
    - behavior_df: pandas DataFrame with columns ['trial_number', 'total_distance']
    - trial_number: int, the trial number to analyze
    
    Returns:
    - A dictionary with statistical measures for the specified trial.
    """
    trial_data = behavior_df[behavior_df['trial'] == trial_number][attribute_name]
    analysis_results = {
        'mean': trial_data.mean(),
        'median': trial_data.median(),
        'std_dev': trial_data.std(),
        'variance': trial_data.var(),
        'min': trial_data.min(),
        'max': trial_data.max(),
        'quartiles': trial_data.quantile([0.25, 0.5, 0.75]).to_dict()
    }
    return analysis_results

def pprint_single_trail_stats(behavior_df, trial_number, attribute_name='total_distance'):
    single_trial_stats = analyze_single_trial(behavior_df, trial_number, attribute_name)

    print(f"Statistics for {attribute_name}, trial {trial_number}:\n{json.dumps(single_trial_stats, indent=4)}")

def compare_multiple_trials(behavior_df, trial_numbers, attribute_name):
    """
    Compare multiple trials and provide overall statistics from a DataFrame.
    
    Parameters:
    - behavior_df: pandas DataFrame with columns ['trial_number', 'total_distance']
    - trial_numbers: list of int, the trial numbers to compare
    
    Returns:
    - A dictionary with overall statistics and ANOVA test result if more than two trials are compared.
    """
    trials_data = behavior_df[behavior_df['trial'].isin(trial_numbers)][attribute_name]
    overall_stats = {
        'overall_mean': trials_data.mean(),
        'overall_std_dev': trials_data.std(),
        'overall_variance': trials_data.var(),
    }
    
    if len(trial_numbers) > 2:
        # Perform ANOVA test to compare the means of the trials
        grouped_data = [group[attribute_name].values for name, group in behavior_df.groupby('trial') if name in trial_numbers]
        f_value, p_value = stats.f_oneway(*grouped_data)
        overall_stats['anova_test'] = {'f_value': f_value, 'p_value': p_value}
    
    return overall_stats

def pprint_multiple_trail_stats(behavior_df, trial_numbers, attribute_name='total_distance'):
    comparison_results = compare_multiple_trials(behavior_df, trial_numbers, attribute_name)
    
    print(f"Comparison of {attribute_name} across {len(trial_numbers)} trials:\n" + json.dumps(comparison_results, indent=4))


def plot_trial_attb_extrema(behavior_data_df, attribute_name='position', filter_outliers=False):
    max_positions = behavior_data_df.groupby('trial')[attribute_name].max()
    min_positions = behavior_data_df.groupby('trial')[attribute_name].min()

    if filter_outliers:
        # Identify outliers in max positions
        Q1 = max_positions.quantile(0.25)
        Q3 = max_positions.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold_high = Q3 + 1.5 * IQR
        outlier_threshold_low = Q1 - 1.5 * IQR

        # Filter out trials that are outliers in terms of max position
        filtered_max_positions = max_positions[(max_positions <= outlier_threshold_high) & (max_positions >= outlier_threshold_low)]
        filtered_min_positions = min_positions[filtered_max_positions.index]
    else:
        filtered_max_positions = max_positions
        filtered_min_positions = min_positions

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    filtered_max_positions.plot(kind='bar', color='blue', label='Max Position')
    filtered_min_positions.plot(kind='bar', color='red', label='Min Position', alpha=0.5)
    plt.title(f'Max and Min {attribute_name} for Each Trial (Excluding Outliers)')
    plt.xlabel('Trial Number')
    plt.ylabel(f'{attribute_name}')
    plt.xticks(rotation=90)  # Rotate labels to avoid overlap
    plt.legend()
    plt.show()