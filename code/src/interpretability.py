from captum.attr import IntegratedGradients, Saliency, InputXGradient, NoiseTunnel
#from tqdm.notebook import tqdm
import torch
import xarray as xr
import matplotlib.pyplot as plt
from src.plots import plot_basemap
import scipy as sp 
from src.utils import log_transform, norm_transform, norm_minus1_to_plus1_transform
import scipy.ndimage
import numpy as np

class Interpretability():
    
    def __init__(self,
                 model,
                 test_period=None):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = self.model.to(self.device)
        self.ig = IntegratedGradients(net)
        self.nt = NoiseTunnel(self.ig)
        if test_period is not None:
            self.test_period = test_period
        else:
            self.test_period = ('2001', '2014')
        self.train_period = ('1950', '2000')
        print(f'running on {self.device}')
        self.epsilon = 0.0001
        self.baseline = None
        self.num_latitudes = 60
        self.num_longitudes = 96
        
        fname = '/data/era5.nc'
        self.era5 = xr.open_dataset(fname).era5_precipitation*3600*24
        self.era5 = self.era5.sel(time=slice(self.test_period[0], self.test_period[1]))
        
        fname = '/data/poem_historical.nc'
        self.poem_base = xr.open_dataset(fname).precipitation*3600*24
        self.poem = self.poem_base.sel(time=slice(self.test_period[0], self.test_period[1]))
        self.poem_reference = self.poem_base.sel(time=slice(self.train_period[0], self.train_period[1]))


    def run(self,
            dataset_name:str,
            attribution_method='integrated_gradients',
            baseline=None,
            ):

        self.baseline = baseline

        self.dataset = getattr(self, dataset_name)
        self.attribution_method = attribution_method 

        if baseline is None and self.attribution_method is 'integrated_gradients':
            self.baseline = 'zeros'

        print(f'Using attribution method: {self.attribution_method}')
        if baseline is not None:
            print(f'Using baseline: {self.baseline}')

        results = torch.zeros(self.dataset.shape)
        reference = log_transform(self.poem_reference, self.epsilon)
        input_data = []

        for i in range(len(self.dataset.time)):
            self.data = self.dataset.isel(time=i)

            self.data = log_transform(self.data, self.epsilon)

            self.data = norm_minus1_to_plus1_transform(self.data, reference)

            input = torch.from_numpy(self.data.values).unsqueeze(0).unsqueeze(0).to(self.device).float()

            results[i] =  self.interpret(input)
            input_data.append(input)

        input_data = torch.cat(input_data)
        results = self.convert_to_netcdf(results)
        return results, input_data


    def get_baseline(self, input):

        shape = (1,1,self.num_latitudes,self.num_longitudes)

        if self.baseline is 'zeros':
            baseline = torch.zeros(shape).to(self.device)

        if self.baseline is 'shuffle':
            baseline = input.flatten().cpu().numpy()
            np.random.shuffle(baseline)
            baseline = baseline.reshape(shape)
            baseline = torch.from_numpy(baseline).reshape(shape).to(self.device)

        if self.baseline is 'blurred':
            baseline = sp.ndimage.gaussian_filter(input.squeeze().cpu().numpy(), sigma=4)
            baseline = torch.from_numpy(baseline).reshape(shape).to(self.device)
        return baseline


    def interpret(self, input):
        shape = (1,1,self.num_latitudes,self.num_longitudes)
        tmp = torch.zeros(shape).to(self.device)
        if self.baseline is not None:
            baseline = self.get_baseline(input)
        for i in range(0,4):
            for j in range(0,9):
                if self.attribution_method == 'integrated_gradients':
                    attributions, _ = self.ig.attribute(input,
                                                        target=(0,i,j),
                                                        baselines=baseline,
                                                        return_convergence_delta=True)
                if self.attribution_method == 'smooth_grad':
                    attributions = self.nt.attribute(input,
                                                     target=(0,i,j),
                                                     nt_type='smoothgrad',
                                                     nt_samples=10)
                tmp += attributions
        tmp *= 1.0/tmp.max()
        return tmp
        
        
    def convert_to_netcdf(self, result):
        result = result.detach().cpu()
        result = xr.DataArray(
                    data= result.squeeze(0),
                    dims=["time", "latitude", "longitude"],
                    coords=dict(
                        time=self.dataset.time.values.reshape(len(self.dataset.time)),
                        latitude=self.dataset.latitude,
                        longitude=self.dataset.longitude,
                    ),
                    attrs=dict(
                        description="Integrated gradients"
                    ))
        return result.to_dataset(name='integrated_gradients')
    
    
    def get_era5(self):
        return self.era5
    
    
    def get_poem(self):
        return self.poem
    

def plot_attribution(xai: xr.DataArray,
                     poem: xr.DataArray,
                     era5: xr.DataArray,
                     fname=None,
                     input=None,
                     average=True,
                     low_pass_filter=True,
                     vmin=-0.025,
                     vmax=0.025):
    
    if average:
        xai = xai.mean(dim='time')
        poem = poem.mean(dim='time')
        era5 = era5.mean(dim='time')

    if low_pass_filter:
        result = sp.ndimage.gaussian_filter(xai, sigma=1.5)
    else:
        result = xai
    
    result = xr.DataArray(
                        data=result,
                        dims=["latitude", "longitude"],
                        coords=dict(latitude=poem.latitude, longitude=poem.longitude)
                        )

    if input is not None:
        plt.figure(figsize=(23,8))
        input = xr.DataArray(
                        data=input.squeeze(),
                        dims=["latitude", "longitude"],
                        coords=dict(latitude=poem.latitude, longitude=poem.longitude)
                        )
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        plt.rcParams.update({'font.size': 12})
        offset = 10
    

    plot_basemap(poem, 'Mean precipitation [mm/d]', 0, 10, 0.85, 'viridis_r',
                 cbar_extend='max',
                 projection='robin',\
                 contours=result,
                 cbar_position='left',
                 fig=fig, 
                 draw_coordinates=True, 
                 contourf=True)

    if input is not None:
        plt.subplot(134+offset)
        plt.title('Discriminator raw input')
        plot_basemap(input, 'normalized input', -2, 2, 0.8, 'seismic',
                     cbar_extend='both', contours=False)

    if fname is not None:    
        plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        print(fname) 
    plt.show()
