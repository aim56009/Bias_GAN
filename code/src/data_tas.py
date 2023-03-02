from dataclasses import dataclass
import cftime
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader
from Bias_GAN.code.src.utils import (log_transform,
                       norm_minus1_to_plus1_transform, norm_transform)


@dataclass
class TestData():
    
    era5: xr.DataArray
    gan: xr.DataArray
    climate_model: xr.DataArray = None
    uuid: str = None
    model = None


    def model_name_definition(self, key):
        dict = {
            'era5': 'ERA5',
            'gan': 'GAN (unconstrained)',
            'climate_model': 'Climate model',         
        }
        return dict[key]


    def colors(self, key):
        dict = {
            'era5': 'k',
            'gan': 'brown',
            'climate_model': 'r',
        }
        return dict[key]

        
    def convert_units(self):
        """ from mm/s to mm/d"""
        self.climate_model = self.climate_model
        self.era5 = self.era5
        self.gan = self.gan

    
    def crop_test_period(self):
        print('')
        print(f'Test set period: {self.gan.time[0].values} - {self.gan.time[-1].values}')
        self.climate_model = self.climate_model.sel(time=slice(self.gan.time[0], self.gan.time[-1]))
        self.era5 = self.era5.sel(time=slice(self.gan.time[0], self.gan.time[-1]))

        
    def show_mean(self):
        print('')
        print(f'Mean [K]:')
        print(f'ERA5: {self.era5.mean().values:2.3f}')
        print(f'Climate Model: {self.climate_model.mean().values:2.3f}')
        print(f'GAN:  {self.gan.mean().values:2.3f}')




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

        #if 'poem_precipitation' in climate_model.variables:
        #    climate_model =  climate_model.poem_precipitation
        #else:
        climate_model =  climate_model.tas

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
                               .tas

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
            climate_model =  climate_model.tas
        else:
            climate_model =  climate_model.tas

        if not self.config.lazy:
            climate_model = climate_model.load()

        climate_model = climate_model.sel(time=slice(str(self.config.train_start), str(self.config.train_end)))

        return climate_model


    def load_climate_model_data(self):

        climate_model = xr.open_dataset(self.config.projection_path)

        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.tas
        else:
            climate_model =  climate_model.tas

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
