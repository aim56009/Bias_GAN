from dataclasses import dataclass
import cftime
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader
from src.utils import (log_transform,
                       norm_minus1_to_plus1_transform, norm_transform)


@dataclass
class TestData():
    
    era5: xr.DataArray
    gan: xr.DataArray
    climate_model: xr.DataArray = None
    cmip_model: xr.DataArray = None
    gan_constrained: xr.DataArray = None
    poem: xr.DataArray = None
    quantile_mapping: xr.DataArray = None
    uuid: str = None
    model = None


    def model_name_definition(self, key):
        dict = {
            'era5': 'ERA5',
            'gan': 'GAN (unconstrained)',
            'climate_model': 'Climate model',
            'cmip_model': 'GFDL-ESM4',
            'poem': 'CM2Mc-LPJmL',
            'gan_constrained': 'GAN',
            'quantile_mapping': 'Quantile mapping',
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'era5': 'k',
            'gan': 'brown',
            'cmip_model': 'b',
            'poem': 'r',
            'gan_constrained': 'c',
            'quantile_mapping': 'm',
        }
        return dict[key]

        
    def convert_units(self):
        """ from mm/s to mm/d"""
        self.climate_model = self.climate_model*3600*24
        self.era5 = self.era5*3600*24
        self.gan = self.gan*3600*24

    
    def crop_test_period(self):
        print('')
        print(f'Test set period: {self.gan.time[0].values} - {self.gan.time[-1].values}')
        self.climate_model = self.climate_model.sel(time=slice(self.gan.time[0], self.gan.time[-1]))
        self.era5 = self.era5.sel(time=slice(self.gan.time[0], self.gan.time[-1]))

        
    def show_mean(self):
        print('')
        print(f'Mean [mm/d]:')
        print(f'ERA5: {self.era5.mean().values:2.3f}')
        print(f'Climate Model: {self.climate_model.mean().values:2.3f}')
        print(f'GAN:  {self.gan.mean().values:2.3f}')

@dataclass
class CMIP6Data():
    
    era5: xr.DataArray = None
    gan_constrained: xr.DataArray = None
    poem: xr.DataArray = None
    mpi: xr.DataArray = None
    cesm2: xr.DataArray = None
    gfdl: xr.DataArray = None


    def model_name_definition(self, key):
        dict = {
            'era5': 'ERA5',
            'gan_constrained': 'GAN',
            'poem': 'CM2Mc-LPJmL',
            'mpi': 'MPI-ESM1-2-HR',
            'gfdl': 'GFDL-ESM4',
            'cesm2': 'CESM2',
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'era5': 'k',
            'poem': 'tab:orange',
            'gfdl': 'tab:blue',
            'mpi': 'tab:red',
            'cesm2': 'tab:green',
            'gan_constrained': 'tab:red',
        }
        return dict[key]
        

def load_cmip6_model(fname: str, historical_test_period: list) -> xr.DataArray:
    data = xr.open_dataset(fname).precipitation*3600*24
    data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
    #data = slice_time(data, int(historical_test_period[0]), int(historical_test_period[1]))
    return data 


def slice_time(dataset, year_start, year_end):
    year_start = np.where(dataset.time.values == cftime.DatetimeNoLeap(year_start,1,1,12))[0][0]
    year_end = np.where(dataset.time.values == cftime.DatetimeNoLeap(year_end,12,31,12))[0][0]

    dataset = dataset.isel(time=slice(year_start, year_end))
    return dataset


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 config,
                 training_batch_size: int = 4,
                 test_batch_size: int = 64):


        super().__init__()

        self.config = config
        self.training_batch_size = training_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            self.train = CycleDataset('train', self.config)
            self.valid = CycleDataset('valid', self.config)

        if stage == 'test':
            self.test = CycleDataset('test', self.config)

        if stage == 'predict':
            self.test = ProjectionDataset(self.config)


    def train_dataloader(self):
        return DataLoader(self.train,
                         batch_size=self.training_batch_size,
                         shuffle=True,
                         num_workers=0,
                         pin_memory=True)


    def val_dataloader  (self):
        return DataLoader(self.valid,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)


    def test_dataloader (self):
        return DataLoader(self.test,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)


class CycleDataset(torch.utils.data.Dataset):
    
    def __init__(self, stage, config, epsilon=0.0001):
        """ 
            stage: train, valid, test
        """
        self.transforms = config.transforms
        self.epsilon = epsilon
        self.config = config

        if config.lazy:
            self.cache = False
            self.chunks = {'time': 1}
        else:
            self.cache = True
            self.chunks = None

        self.splits = {
                "train": [str(config.train_start), str(config.train_end)],
                "valid": [str(config.valid_start), str(config.valid_end)],
                "test":  [str(config.test_start), str(config.test_end)],
        }

        self.stage = stage
        self.climate_model = self.load_climate_model_data()
        climate_model_reference = self.load_climate_model_data(is_reference=True)
        self.era5 = self.load_era5_data()
        era5_reference = self.load_era5_data(is_reference=True)
        self.num_samples = len(self.era5.time.values)
        self.era5 = self.apply_transforms(self.era5, era5_reference)
        self.climate_model = self.apply_transforms(self.climate_model, climate_model_reference)


    def load_climate_model_data(self, is_reference=False):
        """ Y-domain samples """

        climate_model = xr.open_dataset(self.config.poem_path,
                                        cache=self.cache, chunks=self.chunks)

        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.poem_precipitation
        else:
            climate_model =  climate_model.precipitation

        if not self.config.lazy:
            climate_model = climate_model.load()

        if is_reference:
            climate_model = climate_model.sel(time=slice(self.splits['train'][0],
                                                         self.splits['train'][1]))
        else:
            climate_model = climate_model.sel(time=slice(self.splits[self.stage][0],
                                                         self.splits[self.stage][1]))

        return climate_model


    def load_era5_data(self, is_reference=False):
        """ X-domain samples """

        era5 = xr.open_dataset(self.config.era5_path,
                               cache=self.cache, chunks=self.chunks)\
                               .era5_precipitation

        if not self.config.lazy:
            era5 = era5.load()

        if is_reference:
            era5 = era5.sel(time=slice(self.splits['train'][0],
                                       self.splits['train'][1]))
        else:
            era5 = era5.sel(time=slice(self.splits[self.stage][0],
                                 self.splits[self.stage][1]))

        return era5
        

    def apply_transforms(self, data, data_ref):

        if 'log' in self.transforms:
            data = log_transform(data, self.epsilon)
            data_ref = log_transform(data_ref, self.epsilon)

        if 'normalize' in self.transforms:
            data = norm_transform(data, data_ref)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data = norm_minus1_to_plus1_transform(data, data_ref)
        
        return data


    def __getitem__(self, index):

        x = torch.from_numpy(self.era5.isel(time=index).values).float().unsqueeze(0)
        y = torch.from_numpy(self.climate_model.isel(time=index).values).float().unsqueeze(0)

        sample = {'A': x, 'B': y}
        
        return sample

    def __len__(self):
        return self.num_samples


class ProjectionDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, epsilon=0.0001):
        """ 
            Test dataset for CMIP6 projections. Only returns the inputs y.
        """

        self.transforms = config.transforms
        self.epsilon = epsilon
        self.config = config
        self.climate_model = self.load_climate_model_data()
        climate_model_reference = self.load_climate_model_reference_data()
        self.num_samples = len(self.climate_model.time.values)
        self.climate_model = self.apply_transforms(self.climate_model, climate_model_reference)


    def load_climate_model_reference_data(self):

        climate_model = xr.open_dataset(self.config.poem_path)

        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.poem_precipitation
        else:
            climate_model =  climate_model.precipitation

        if not self.config.lazy:
            climate_model = climate_model.load()

        climate_model = climate_model.sel(time=slice(str(self.config.train_start), str(self.config.train_end)))

        return climate_model


    def load_climate_model_data(self):

        climate_model = xr.open_dataset(self.config.projection_path)

        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.poem_precipitation
        else:
            climate_model =  climate_model.precipitation

        if not self.config.lazy:
            climate_model = climate_model.load()

        return climate_model


    def apply_transforms(self, data, data_ref):

        if 'log' in self.transforms:
            data = log_transform(data, self.epsilon)
            data_ref = log_transform(data_ref, self.epsilon)

        if 'normalize' in self.transforms:
            data = norm_transform(data, data_ref)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data = norm_minus1_to_plus1_transform(data, data_ref)
        
        return data


    def __getitem__(self, index):

        y = torch.from_numpy(self.climate_model.isel(time=index).values).float().unsqueeze(0)

        return {'B': y}


    def __len__(self):
        return self.num_samples


