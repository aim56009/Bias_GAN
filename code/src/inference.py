import os
import xarray as xr
import torch
#from tqdm import tqdm
from IPython.display import Image, display

from src.model import CycleGAN, DataModule, ConstrainedGenerator
from src.data import TestData, CycleDataset, ProjectionDataset,  load_cmip6_model, CMIP6Data
from src.plots import PlotAnalysis
from src.utils import log_transform, inv_norm_transform, inv_log_transform, inv_norm_minus1_to_plus1_transform, norm_minus1_to_plus1_transform, config_from_file
from main import Config
from src.quantile_mapping import QuantileMapping
from src.projection_utils import ProjectionPreparation
from src.xarray_utils import write_dataset


class Inference():

    """ Execute model on test data and return output as NetCDF. """
    
    def __init__(self,
                 config,
                 constrain=False,
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
        self.epsilon = config.epsilon
        self.projection = projection
        self.projection_path = projection_path

        self.model = None
        self.model_output = None
        self.dataset = None

        self.transforms = config.transforms
        self.max_num_inference_steps = max_num_inference_steps
        self.tst_batch_sz = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    def load_model(self, checkpoint_path):
    
        model = CycleGAN().load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.freeze()
        self.model = model.to(self.device)
        self.model = ConstrainedGenerator(self.model.g_B2A, constrain=self.constrain)


    def get_model(self):
        return self.model 

        
    def get_dataloader(self):

        datamodule = DataModule(self.config,
                                trn_batch_sz = 1,
                                tst_batch_sz = self.tst_batch_sz)

        if self.projection:
            print('running projection')
            datamodule.setup("predict")
        else:
            datamodule.setup("test")
        dataloader = datamodule.test_dataloader()

        return dataloader

    def get_projection_dataloader(self):

        dataloader = ProjectionDataset(self.config)
        return dataloader
    
        
    def compute(self):
        """ Use B (ESM) -> A (ERA5) generator for inference """

        test_data = self.get_dataloader()

        data = []

        print("Start inference:")
        for idx, sample in enumerate(test_data):
            sample = sample['B'].to(self.device)
            yhat = self.model(sample)

            data.append(yhat.squeeze().cpu())
            if self.max_num_inference_steps is not None:
                if idx > self.max_num_inference_steps - 1:
                    break
            
        self.model_output = torch.cat(data)


    def test(self):
        dataset = CycleDataset('test', self.config)
        test_data = dataset[0]
        sample = test_data['A'][0]
        data = self.inv_transform(sample)
        print(data.min(), data.max())

    
    def get_netcdf_result(self):
        
        time = self.poem.sel(time=slice(self.test_start, self.test_end)).time

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
                longitude=longitude,
            ),
            attrs=dict(
                description="gan_precipitation",
                units="mm/s",
            ))
        
        gan_dataset = gan_data.to_dataset(name="gan_precipitation")
        self.gan_dataset = gan_dataset.transpose('time', 'latitude', 'longitude')

        return self.gan_dataset


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
        
        ds = self.get_netcdf_result()
        path  = self.results_path + fname
        ds.to_netcdf(path)


class EvaluateCheckpoints():
    """ 
        Interate over model checkpoints and
        show the test set results.
    """
    
    def __init__(self,
                 checkpoint_path,
                 plot_summary=False,
                 show_plots=False,
                 save_model=False,
                 constrain=False,
                 epoch_index=None,
                 projection=False,
                 max_num_inference_steps=None,
                 projection_path=None
                 ):

        self.checkpoint_path = checkpoint_path
        print(f'loading checkpoints from directory: {self.checkpoint_path}')
        self.config_path = Config.config_path
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
        self.test_data = None
        self.constrain = constrain
        self.epoch_index = epoch_index
        self.max_num_inference_steps = max_num_inference_steps


    def load_config(self):
        path = self.checkpoint_path
        self.uuid = self.get_uuid_from_path(path)
        config = config_from_file(f'{self.config_path}config_model_{self.uuid}.json')
        if self.projection_path is not None:
            config.projection_path = self.projection_path
        return config


    def get_uuid_from_path(self, path: str):
        import re
        uuid4hex = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', re.I)
        uuid = uuid4hex.search(path).group(0)
        return uuid


    def run(self):
        
        self.config = self.load_config()
        
        files = self.get_files(self.checkpoint_path)

        if self.epoch_index is not None:
            files = [files[self.epoch_index-1]]

        for i, fname in enumerate(files):
            self.checkpoint_idx = i+1
            self.num_checkpoints = len(files)
            print(f'Checkpoint {self.checkpoint_idx} / {self.num_checkpoints}:')
            print(fname)
            print('')
            self.run_inference(fname)
            self.read_test_data()
            self.get_plots()
            
        return self.get_test_data()
        
        
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
                        max_num_inference_steps=self.max_num_inference_steps)
        inf.load_model(path)
        inf.compute()
        self.gan_results = inf.get_netcdf_result()
        self.model = inf.get_model()
        if self.save_model:
            inf.write(self.model_fname)
    
        
    def read_test_data(self):
    
        climate_model = xr.open_dataset(self.config.poem_path)
        if 'poem_precipitation' in climate_model.variables:
            climate_model =  climate_model.poem_precipitation
        else:
            climate_model =  climate_model.precipitation
        era5 = xr.open_dataset(self.config.era5_path).era5_precipitation
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


class LoadData():

    def __init__(self,
        gan_path,
        config
        ):

        self.config = config
        self.gan_path = gan_path
        self.test_period = self.config.test_period[0]
        self.unit_conversion = 3600*24
        self.config.run_models = False
                
    def get_gan_output(self,
                       fname_netcdf,
                       constrain=True
                      ):

        path = self.gan_path

        if self.config.run_models:
            gan_output = EvaluateCheckpoints(path,
                                       constrain=constrain,
                                       plot_summary=False,
                                       epoch_index=self.config.epoch_index,\
                                       max_num_inference_steps=None,
                                       projection=self.config.projection,\
                                       projection_path=self.config.projection_path).run()
            write_dataset(gan_output.gan, fname_netcdf)
            gan_output = gan_output.gan.to_dataset(name='gan_precipitation')

        else:
             gan_output = xr.open_dataset(fname_netcdf)
        gan_output = gan_output.gan_precipitation.rename('precipitation')
        gan_output = self.test_set_crop(gan_output)
        return gan_output


    def test_set_crop(self, data):
        data = data.sel(time=slice(self.test_period[0],
                                   self.test_period[1]))
        return data


    def get_poem_output(self):
        poem = xr.open_dataset(self.config.fname_poem).precipitation*self.unit_conversion
        poem = self.test_set_crop(poem)
        return poem


    def get_era5_data(self):
        era5 = xr.open_dataset(self.config.era5_path).era5_precipitation*self.unit_conversion
        era5 = self.test_set_crop(era5)
        return era5


    def get_quantile_mapping_output(self):

        if self.config.run_models:
            qm = QuantileMapping(self.config.fname_poem,
                                 train_split=['1990', '2000'],
                                 test_split=['2001', '2017'],
                                 num_quantiles=1000)
            qm.run()
            qm_data = qm.get_test_data()
            qm_data = qm_data.rename('precipitation')

            qm_data = self.test_set_crop(qm_data)
            write_dataset(qm_data, self.config.fname_quantile_mapping)
        else:    
            qm_data = xr.open_dataset(self.config.fname_quantile_mapping).precipitation

        return qm_data


    def remove_leap_year(self, data):
        data = data.sel(time=~((data.time.dt.month == 2) & (data.time.dt.day == 29)))
        data = data.sel(time=slice(self.test_period[0], self.test_period[1])).to_dataset()
        return data


    def get_cmip6_output(self):
        data = {
            'gfdl': load_cmip6_model(self.config.fname_gfdl, self.test_period),
            'mpi':  load_cmip6_model(self.config.fname_mpi, self.test_period),
            'cesm2': load_cmip6_model(self.config.fname_cesm, self.test_period)
        }
        return data


    def collect_historical_data(self):

        gan_constrained = self.get_gan_output(self.config.fname_gan_constrained, 
                                              constrain=True) 

        gan_unconstrained = self.get_gan_output(self.config.fname_gan_unconstrained, 
                                              constrain=False) 

        poem = self.get_poem_output()

        era5 = self.get_era5_data()

        qm = self.get_quantile_mapping_output()

        cmip = self.get_cmip6_output()

        test_data = TestData(era5,
                             gan_unconstrained,
                             cmip_model=cmip['gfdl'],
                             gan_constrained=gan_constrained,
                             poem=poem,
                             quantile_mapping=qm)

        return test_data


    def collect_historical_cmip_data(self):

        gan_constrained = self.get_gan_output(self.config.fname_gan_constrained, 
                                              constrain=True) 

        poem = self.get_poem_output()

        era5 = self.get_era5_data()

        cmip = self.get_cmip6_output()

        test_data = CMIP6Data(era5=era5,
                            poem=poem,
                            gan_constrained=gan_constrained,
                            gfdl=cmip['gfdl'],
                            mpi=cmip['mpi'],
                            cesm2=cmip['cesm2'])

        return test_data


    def collect_projection_data(self, latitude_bounds=None, 
                                names=['gan', 'gan_constrained', 'poem', 'cmip_model']):

        gan_constrained = self.get_gan_output(self.config.fname_gan_constrained, 
                                              constrain=True) 

        gan_unconstrained = self.get_gan_output(self.config.fname_gan_unconstrained, 
                                              constrain=False) 

        poem = self.get_poem_output()

        cmip = self.get_cmip6_output()
        gfdl = cmip['gfdl']
        gfdl = self.remove_leap_year(gfdl)
        if type(gfdl) is xr.Dataset:
            gfdl = gfdl.precipitation

        test_data = TestData(gan_unconstrained,
                             gan_unconstrained,
                             cmip_model=gfdl,
                             gan_constrained=gan_constrained,
                             poem=poem)

        _ = ProjectionPreparation(test_data,
                           names ,
                           self.test_period,
                           latitude_bounds=latitude_bounds).run() 
        return test_data



    def collect_projection_cmip_data(self, latitude_bounds=None):

        gan_constrained = self.get_gan_output(self.config.fname_gan_constrained, 
                                              constrain=True) 



        poem = self.get_poem_output()

        cmip = self.get_cmip6_output()
        gfdl = cmip['gfdl']
        gfdl = self.remove_leap_year(gfdl)
        if type(gfdl) is xr.Dataset:
            gfdl = gfdl.precipitation

        test_data = CMIP6Data(era5=None,
                 poem=poem,
                 gan_constrained=gan_constrained,
                 gfdl=gfdl,
                 mpi=cmip['mpi'],
                 cesm2=cmip['cesm2'])

        _ = ProjectionPreparation(test_data,
                                   ['poem', 'gfdl', 'mpi', 'cesm2'],
                                    self.test_period,
                                    latitude_bounds=latitude_bounds).run() 
        return test_data



def create_folder(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
