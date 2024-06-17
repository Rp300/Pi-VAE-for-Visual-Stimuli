import os
import sys

def get_project_path():
    """Return the absolute path to the parent directory of the project."""
    return "/Users/rp/Desktop/Research/CN^3/Thesis Material/2-Region-Latent-Alignment"

def get_data_dir():
    """Return the absolute path to the data directory."""
    project_path = get_project_path()
    return os.path.join(project_path, "data")

def get_src_dir():
    """Return the absolute path to the src directory."""
    project_path = get_project_path()
    return os.path.join(project_path, "src")

def get_models_dir():
    """Return the absolute path to the models directory."""
    project_path = get_project_path()
    return os.path.join(project_path, "models")

def get_plots_dir():
    """Return the absolute path to the plots directory."""
    project_path = get_project_path()
    return os.path.join(project_path, "plots")

def add_project_dirs_to_path():
    """Add project root and src directory to Python path if they are not already present."""
    # from .file_utils import get_project_path, get_src_dir  # Importing locally to avoid circular import
    
    # Get project and source directory paths
    path_to_project = get_project_path()
    path_to_src = get_src_dir()
    
    # Add src directory and project root to Python path if they are not already present
    if path_to_src not in sys.path:
        sys.path.append(path_to_src)

    if path_to_project not in sys.path:
        sys.path.append(path_to_project)