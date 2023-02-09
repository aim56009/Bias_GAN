import os
import xarray as xr
import torch
from IPython.display import Image, display

from Bias_GAN.code.src.model import CycleGAN, DataModule, ConstrainedGenerator
from Bias_GAN.code.src.plots import PlotAnalysis
from Bias_GAN.code.src.utils import log_transform, inv_norm_transform, inv_log_transform, inv_norm_minus1_to_plus1_transform, norm_minus1_to_plus1_transform, config_from_file

from dataclasses import dataclass
import cftime
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader
from Bias_GAN.code.src.utils import (log_transform,
                       norm_minus1_to_plus1_transform, norm_transform)

class Inference():

    """ Execute model on test data and return output as NetCDF. """
    
    def __init__(self,
                 config,
                 constrain=False,
                 validation=False,
                 projection=False,
                 projection_path=None,
                 max_num_inference_steps=None):
        

        self.config = config
        self.constrain = constrain
        self.results_path = config.results_path

        self.poem = xr.open_dataset(self.config.poem_path)
        self.era5 = xr.open_dataset(self.config.era5_path)

        self.train_start = str(config.train_start)
        self.train_end = str(config.train_end)
        self.test_start = str(config.test_start)
        self.test_end = str(config.test_end)
        
        self.validation = validation
        if self.validation==True:
          self.valid_start = str(config.valid_start)
          self.valid_end = str(config.valid_end)

        self.epsilon = config.epsilon
        self.projection = projection
        self.projection_path = projection_path

        self.model = None
        self.model_output = None
        self.dataset = None
        
        self.reconstruct_model = None
        self.model_output_reconstr = None

        self.transforms = config.transforms
        self.max_num_inference_steps = max_num_inference_steps
        self.tst_batch_sz = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    def load_model(self, checkpoint_path):
    
        model = CycleGAN().load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.freeze()
        self.model = model.to(self.device)
        self.model = ConstrainedGenerator(self.model.g_B2A, constrain=self.constrain)

        model_ = CycleGAN().load_from_checkpoint(checkpoint_path=checkpoint_path)
        model_.freeze()
        self.reconstruct_model = model_.to(self.device)
        self.reconstruct_model = ConstrainedGenerator(self.reconstruct_model.g_A2B, constrain=self.constrain)


    def get_model(self):
        return self.model , self.reconstruct_model

        
    def get_dataloader(self):
        datamodule = DataModule(self.config,
                                training_batch_size = 1,
                                test_batch_size = self.tst_batch_sz)
        if self.projection:
            print('running projection')
            datamodule.setup("predict")
        else:
            datamodule.setup("test")
        
        if self.validation==False:
          dataloader = datamodule.test_dataloader()
        else:
          dataloader = datamodule.val_dataloader()

        return dataloader

    def get_projection_dataloader(self):

        dataloader = ProjectionDataset(self.config)
        return dataloader
    
        
    def compute(self):
        """ Use B (ESM) -> A (ERA5) generator for inference """

        data = []
        print("Start inference:")
        if self.validation==True:
          valid_data = self.get_dataloader()
          for idx, sample in enumerate(valid_data):
              sample = sample['B'].to(self.device)
              yhat = self.model(sample)

              data.append(yhat.squeeze().cpu())
              if self.max_num_inference_steps is not None:
                  if idx > self.max_num_inference_steps - 1:
                      break
        else:
          test_data = self.get_dataloader()
          for idx, sample in enumerate(test_data):
              sample = sample['B'].to(self.device)
              yhat = self.model(sample)

              data.append(yhat.squeeze().cpu())
              if self.max_num_inference_steps is not None:
                  if idx > self.max_num_inference_steps - 1:
                      break
            
        self.model_output = torch.cat(data)


    def compute_reconstruction(self):
          """ Use generated  A' (ERA5) -> B (ESM)  for inference """

          data_reconstr = []
          data = []
          print("Start inference:")
          if self.validation==True:
            valid_data = self.get_dataloader()
            for idx, sample in enumerate(valid_data):
                sample = sample['B'].to(self.device)
                yhat = self.model(sample) ### self.g_B2A
                reconstruct = self.reconstruct_model(yhat) ### self.g_A2B
                #plot_reconstruction(reconstruct)
                
                data_reconstr.append(reconstruct.squeeze().cpu())
                data.append(yhat.squeeze().cpu())
                if self.max_num_inference_steps is not None:
                    if idx > self.max_num_inference_steps - 1:
                        break
          else:
            test_data = self.get_dataloader()
            for idx, sample in enumerate(test_data):
                sample = sample['B'].to(self.device)        
                yhat = self.model(sample)  ### self.g_B2A
                reconstruct = self.reconstruct_model(yhat) ### self.g_A2B

                """
                print("plot first sample of batch - original data")             
                #cs = plt.pcolormesh(sample.squeeze().cpu()[0,:,:])
                cs = plt.pcolormesh(self.inv_transform(sample.cpu()).squeeze().cpu()[0,:,:]*3600*24,cmap="Blues")
                plt.colorbar(cs, cax = make_axes_locatable(plt.gca()).append_axes('right', size="1.5%", pad=0.4), extend='max')
                plt.show()
                print("plot first sample of batch - generated data")
                #cs = plt.pcolormesh(yhat.squeeze().cpu()[0,:,:])
                cs = plt.pcolormesh(self.inv_transform(yhat.cpu()).squeeze().cpu()[0,:,:]*3600*24,cmap="Blues")
                plt.colorbar(cs, cax = make_axes_locatable(plt.gca()).append_axes('right', size="1.5%", pad=0.4), extend='max')
                plt.show()
                print("plot first sample of batch - reconstructed data")
                #cs = plt.pcolormesh(reconstruct.squeeze().cpu()[0,:,:])
                cs = plt.pcolormesh(self.inv_transform(reconstruct.cpu()).squeeze().squeeze().cpu()[0,:,:]*3600*24,cmap="Blues")
                plt.colorbar(cs, cax = make_axes_locatable(plt.gca()).append_axes('right', size="1.5%", pad=0.4), extend='max')
                plt.show()
                """
                
                data_reconstr.append(reconstruct.squeeze().cpu())
                data.append(yhat.squeeze().cpu())
                if self.max_num_inference_steps is not None:
                    if idx > self.max_num_inference_steps - 1:
                        break
                        
          self.model_output = torch.cat(data)    
          self.model_output_reconstr = torch.cat(data_reconstr)
          #return ori, gen , back

    def test(self):
        dataset = CycleDataset('test', self.config)
        test_data = dataset[0]
        sample = test_data['A'][0]
        data = self.inv_transform(sample)
        print(data.min(), data.max())

    
    def get_netcdf_result(self): 
        
        if self.validation==False:
            print("MODE: TESTING")
            time = self.poem.sel(time=slice(self.test_start, self.test_end)).time
        if self.validation==True:
            print("MODE: VALIDATION")
            time = self.poem.sel(time=slice(self.valid_start, self.valid_end)).time

        if self.projection:
            time = xr.open_dataset(self.projection_path).time

        if self.max_num_inference_steps is not None:
            time = time.isel(time=slice(0, (self.max_num_inference_steps+1)*self.tst_batch_sz))

        latitude = self.poem.lat
        longitude = self.poem.lon
        
        gan_data= xr.DataArray(
            data=self.model_output,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=time,
                latitude=latitude,
                longitude=longitude,),
            attrs=dict(description="gan_precipitation",units="mm/s",))
        
        ### for reconstruction ###
        gan_reconstruct= xr.DataArray(
            data=self.model_output_reconstr,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=time,
                latitude=latitude,
                longitude=longitude,),
            attrs=dict(description="reconstruction_precipitation",units="mm/s",))

        gan_reconstr_dataset = gan_reconstruct.to_dataset(name="gan_reconstruct")
        self.gan_reconstr_dataset = gan_reconstr_dataset.transpose('time', 'latitude', 'longitude')

        gan_dataset = gan_data.to_dataset(name="gan_precipitation")
        self.gan_dataset = gan_dataset.transpose('time', 'latitude', 'longitude')

        return self.gan_dataset, self.gan_reconstr_dataset


    def inv_transform(self, data, reference=None):
        """ The output equals ERA5, therefore it needs to be
            constraind with respect to it
        """
        if reference is None:
            reference = self.era5.era5_precipitation.sel(time=slice(self.train_start, self.train_end)).values

        if 'log' in self.transforms:
            reference = log_transform(reference, self.epsilon)

        if 'normalize' in self.transforms:
            data = inv_norm_transform(data, reference)

        if 'normalize_minus1_to_plus1' in self.transforms:
            data = inv_norm_minus1_to_plus1_transform(data, reference)

        if 'log' in self.transforms:
            data = inv_log_transform(data, self.epsilon)

        return data

    
    def write(self, fname):
        
        ds, ds_reconstr = self.get_netcdf_result()
        path  = self.results_path + fname
        ds.to_netcdf(path)
        ds.to_netcdf(path + "reconstruction")

        

class EvaluateCheckpoints():
    """ 
        Interate over model checkpoints and
        show the test set results.
    """
    
    def __init__(self,
                 checkpoint_path,
                 config_path,
                 plot_summary=False,
                 show_plots=False,
                 save_model=True,
                 constrain=False,
                 epoch_index=None,
                 projection=False,
                 max_num_inference_steps=None,
                 projection_path=None,
                 validation=False,
                 version=""
                 ):

        self.checkpoint_path = checkpoint_path
        #print(f'loading checkpoints from directory: {self.checkpoint_path}')
        self.config_path = config_path
        self.reports_path = "/content/gdrive/MyDrive/bias_gan/results/reports/"
        self.projection_path = projection_path
        self.projection = projection
        self.plot_summary = plot_summary
        self.uuid = None
        self.show_plots = show_plots
        self.gan_results = None
        self.save_model = save_model
        self.model_fname = 'gan.nc'
        self.model = None
        self.reconstruct_model = None
        self.test_data = None
        self.constrain = constrain
        self.epoch_index = epoch_index
        self.max_num_inference_steps = max_num_inference_steps
        self.validation = validation
        self.version = version


    def load_config(self):
        path = self.config_path
        config = config_from_file(path)
        if self.projection_path is not None:
            config.projection_path = self.projection_path
        return config



    def get_uuid_from_path(self, path: str):
        import re
        print("path in get_uuid", path)
        uuid4hex = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
        print("uuid4hex",uuid4hex)
        print("uuid4hex search path",uuid4hex.search(path))
        uuid = uuid4hex.search(path).group(0)
        return uuid


    def run(self):         ############## maybe add checkpoint_path as variable ##############
        
        self.config = self.load_config()
        
        files = [self.checkpoint_path]
        for i, fname in enumerate(files):
            self.checkpoint_idx = i+1
            self.num_checkpoints = len(files)
            print(f'Checkpoint {self.checkpoint_idx} / {self.num_checkpoints}:')
            reconstruct_model_data = self.run_inference(fname)
            self.read_test_data()
            self.get_plots()
            
        return self.get_test_data(), reconstruct_model_data
        
        
    def get_files(self, path: str):
        
        if os.path.isfile(path):
            files = []
            files.append(path) 
        else:
            files = os.listdir(path)
            for i, f in enumerate(files):
                files[i] = os.path.join(path, f) 
        return files

    def run_inference(self, path: str):
        
        inf = Inference(self.config,
                        constrain=self.constrain,
                        projection=self.projection,
                        projection_path=self.projection_path,
                        max_num_inference_steps=self.max_num_inference_steps,
                        validation=self.validation)
        inf.load_model(path)
        #inf.compute()
        inf.compute_reconstruction()
        self.gan_results, self.gan_reconstruction = inf.get_netcdf_result()
        self.model, self.reconstruct_model = inf.get_model()
        
        if self.save_model:
            print("saving model to path:",self.version + "/" + self.model_fname)
            inf.write(self.version + "/" + self.model_fname)

        return self.gan_reconstruction
        
    def read_test_data(self):
    
        climate_model = xr.open_dataset(self.config.poem_path)
        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.tas               #######is it nessecary to mulitply with this??########
        else:
            climate_model =  climate_model.tas                         #######is it nessecary to mulitply with this??########
        era5 = xr.open_dataset(self.config.era5_path).tas             #######is it nessecary to mulitply with this??########
        gan = self.gan_results.gan_precipitation


        data = TestData(era5, gan, climate_model=climate_model)
        data.convert_units()
        data.crop_test_period()
        data.show_mean()
        data.uuid = self.uuid
        data.model = self.model
        
        self.test_data = data


    def get_test_data(self):
        return self.test_data


    def show_reports(self, uuid):
        path = f'{self.reports_path}{uuid}/'
        files = self.get_files(path)
        for file in files:
            fig = Image(filename=file)
            display(fig)
        
        
    def get_plots(self):

        if self.plot_summary:
            plot = PlotAnalysis(self.test_data)
            new_dir = f'{self.reports_path}{self.uuid}/'
            create_folder(new_dir)
            fname = f'{new_dir}model_{self.uuid}_number_{self.checkpoint_idx}.png'
            plot.summary(plot_idx=self.checkpoint_idx, 
                         num_plots=self.num_checkpoints,
                         fname=fname, show_plots=self.show_plots)


def create_folder(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

    
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
            self.valid = CycleDataset('valid', self.config)

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


def show_image(image):
    plt.imshow((image.squeeze()))


def get_random_sample(dataset):
    return dataset[np.random.randint(0, len(dataset))]


from dataclasses import dataclass
import cftime
from torch.utils.data import DataLoader


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
            'climate_model': 'r',
            'gan_constrained': 'c',
            'quantile_mapping': 'm',
        }
        return dict[key]

        
    def convert_units(self):
        """ from mm/s to mm/d"""
        self.climate_model = self.climate_model#*3600*24
        self.era5 = self.era5#*3600*24
        self.gan = self.gan#*3600*24

    
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
