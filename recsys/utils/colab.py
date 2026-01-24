"""Utilities for running notebooks in Google Colab."""

import os
import sys
from pathlib import Path


def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment(
    repo_url='https://github.com/yourusername/modern-recommender-systems',
    branch='main',
    install_mode='pip'
):
    """
    Setup the environment for Google Colab.
    
    Args:
        repo_url: GitHub repository URL
        branch: Branch to use (default: 'main')
        install_mode: 'pip' or 'clone' (default: 'pip')
    """
    if not is_colab():
        print("Not running in Colab, skipping setup.")
        return
    
    print("🚀 Setting up Colab environment...")
    
    if install_mode == 'pip':
        # Install directly from GitHub
        os.system(f'pip install -q git+{repo_url}@{branch}')
        print("✓ Package installed from GitHub")
    else:
        # Clone repository for development
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        if not Path(repo_name).exists():
            os.system(f'git clone -q -b {branch} {repo_url}')
            os.chdir(repo_name)
            os.system('pip install -q -e .')
            print(f"✓ Repository cloned and installed: {repo_name}")
        else:
            print(f"✓ Repository already exists: {repo_name}")
    
    print("✓ Colab environment ready!")


def download_data(data_urls, target_dir='/content/data'):
    """
    Download datasets for Colab.
    
    Args:
        data_urls: List of URLs or dict of {filename: url}
        target_dir: Directory to save data (default: '/content/data')
    """
    if not is_colab():
        print("Not running in Colab, skipping data download.")
        return
    
    os.makedirs(target_dir, exist_ok=True)
    
    if isinstance(data_urls, dict):
        for filename, url in data_urls.items():
            output_path = os.path.join(target_dir, filename)
            print(f"Downloading {filename}...")
            os.system(f'wget -q {url} -O {output_path}')
    else:
        for url in data_urls:
            print(f"Downloading {url}...")
            os.system(f'wget -q {url} -P {target_dir}')
    
    print(f"✓ Data downloaded to {target_dir}")


def get_data_path():
    """Return appropriate data path for environment."""
    if is_colab():
        return '/content/data/'
    else:
        # Assume running from notebooks/ directory
        return str(Path('../data').resolve())


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU available: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected.")
            if is_colab():
                print("  Enable GPU: Runtime → Change runtime type → GPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed, cannot check GPU")
        return False