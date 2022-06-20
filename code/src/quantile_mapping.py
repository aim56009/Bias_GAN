import xarray as xr
from xclim import sdba


class QuantileMapping():

    def __init__(self,
                 climate_model_data_path,
                 num_quantiles=50,
                 train_split=['1990', '2000'],
                 test_split=['2001', '2017']
                 ):

        self.climate_model_data_path = climate_model_data_path
        self.era5_data_path = '/p/tmp/hess/scratch/poem-gan/datasets/era5.nc'
        self.train_split = train_split
        self.test_split = test_split
        self.num_quantiles = num_quantiles
        
    def load_data(self):

        climate_model = xr.open_dataset(self.climate_model_data_path).load()
        
        if 'poem_precipitation' in climate_model.variables:
            climate_model = climate_model.poem_precipitation
        else:
            climate_model = climate_model.precipitation

        self.climate_model_historical = climate_model.sel(time=slice(self.train_split[0], self.train_split[1]))
        self.climate_model_simulation = climate_model.sel(time=slice(self.test_split[0], self.test_split[1]))

        self.era5 = xr.open_dataset(self.era5_data_path).era5_precipitation.load()
        self.era5_reference = self.era5.sel(time=slice(self.train_split[0], self.train_split[1]))


    def run(self):

        self.load_data()
        qm = sdba.DetrendedQuantileMapping(nquantiles=self.num_quantiles)
        qm.train(self.era5_reference, self.climate_model_historical)
        self.quantile_mapped_model = qm.adjust(self.climate_model_simulation)


    def get_test_data(self):

        return self.quantile_mapped_model*3600*24