from src.data import TestData
import xarray as xr

class ProjectionPreparation():
    
    def __init__(self,
                 data: TestData,
                 names: list,
                 projection_period: tuple,
                 latitude_bounds=None):
        self.data = data
        self.names = names
        self.projection_period = projection_period
        self.latitude_bounds = latitude_bounds       
        
    def time_crop(self):
        for name in self.names:
            tmp = getattr(self.data, name).sel(time=slice(self.projection_period[0],
                                                          self.projection_period[1]))
            setattr(self.data, name, tmp)
            
            
    def space_crop(self): 
        for name in self.names:
            if len(self.latitude_bounds) == 4:
                tmp = getattr(self.data, name).sel(latitude=slice(self.latitude_bounds[0],
                                                                  self.latitude_bounds[1]))
                tmp2 = getattr(self.data, name).sel(latitude=slice(self.latitude_bounds[2],
                                                                   self.latitude_bounds[3]))
                tmp = xr.merge([tmp, tmp2]).precipitation
            else:
                tmp = getattr(self.data, name).sel(latitude=slice(self.latitude_bounds[0],
                                                                  self.latitude_bounds[1]))


            setattr(self.data, name, tmp)
            setattr(self.data, name, tmp)
                                               
            
    def compute_global_mean(self): 
        for name in self.names:
            tmp = getattr(self.data, name).mean(dim=('latitude', 'longitude'))
            setattr(self.data, name, tmp)
                                               
                                               
    def resample(self, frequency='M'): 
        for name in self.names:
            tmp = getattr(self.data, name).resample(time=frequency).mean()
            setattr(self.data, name, tmp)
                                               
    def rolling_mean(self, window_width=52*3): 
        for name in self.names:
            tmp = getattr(self.data, name).rolling(time=window_width,
                                                   center=True).mean().dropna("time")
            setattr(self.data, name, tmp)
                                               
            
    def run(self):
        self.time_crop()
        if self.latitude_bounds is not None:
            self.space_crop()
        self.compute_global_mean()
        self.resample()
        self.rolling_mean()