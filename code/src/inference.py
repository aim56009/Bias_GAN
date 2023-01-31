import os
import xarray as xr
import torch
#from tqdm import tqdm
from IPython.display import Image, display

from Bias_GAN.code.src.model import CycleGAN, DataModule, ConstrainedGenerator
from Bias_GAN.code.src.data import TestData, CycleDataset, ProjectionDataset,  load_cmip6_model, CMIP6Data
from Bias_GAN.code.src.plots import PlotAnalysis
from Bias_GAN.code.src.utils import log_transform, inv_norm_transform, inv_log_transform, inv_norm_minus1_to_plus1_transform, norm_minus1_to_plus1_transform, config_from_file
from Bias_GAN.code.main import Config
from Bias_GAN.code.src.quantile_mapping import QuantileMapping
from Bias_GAN.code.src.projection_utils import ProjectionPreparation
from Bias_GAN.code.src.xarray_utils import write_dataset


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
                plot_reconstruction(reconstruct)
                
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

        latitude = self.poem.latitude
        longitude = self.poem.longitude
        
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
        self.reports_path = f'{Config.results_path}reports/'
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
            climate_model =  climate_model.poem_precipitation#*3600*24                     #######is it nessecary to mulitply with this??########
        else:
            climate_model =  climate_model.precipitation #*3600*24                         #######is it nessecary to mulitply with this??########
        era5 = xr.open_dataset(self.config.era5_path).era5_precipitation#*3600*24            #######is it nessecary to mulitply with this??########
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
