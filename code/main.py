from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field
from typing import List
import getpass

from src.trainer import train_cycle_gan



@dataclass
class Config:
    """ 
    Training configuration parameters. For model evaluation parameters see
    src/configuration.py.
    """
    
    scratch_path: str = '/results'
    tensorboard_path: str = f'{scratch_path}/'
    checkpoint_path: str = f'{scratch_path}/'
    config_path: str = f'{scratch_path}/'
    poem_path: str = f'/data/poem_historical.nc'
    era5_path: str = f'/data/era5.nc'
    results_path: str = f'{scratch_path}/'
    projection_path: str = None

    train_start: int = 1950
    train_end: int = 1951 # set to 2000 for full run
    valid_start: int = 2001
    valid_end: int = 2004
    test_start: int = 2001
    test_end: int = 2019
    
    model_name: str = 'poem_gan'

    epochs: int = 2 # set to 250 for reproduction
    progress_bar_refresh_rate: int = 1
    train_batch_size: int = 1
    test_batch_size: int = 64
    transforms: List = field(default_factory=lambda: ['log', 'normalize_minus1_to_plus1'])
    rescale: bool = False
    epsilon: float = 0.0001
    lazy: bool = False
    log_every_n_steps: int = 10
    norm_output: bool = True
    running_bias: bool = False


def main():
    _ = train_cycle_gan(Config())

if __name__ == "__main__":
    main()
