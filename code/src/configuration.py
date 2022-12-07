
from main import Config
from dataclasses import dataclass, field

@dataclass
class HistoricalConfig(Config):
    
    fname_gan_constrained = f'/data/poem_gan_constraint_historical.nc'
    fname_gan_unconstrained = f'/data/poem_gan_historical.nc'
    fname_quantile_mapping = f'/data/poem_qm_historical.nc'
    fname_gfdl = '/data/cmip6_gfdl_historical.nc'
    fname_mpi = '/data/cmip6_mpi_historical.nc'
    fname_cesm = '/data/cmip6_cesm2_historical.nc'
    fname_poem = '/data/pr_gfdl-esm4_historical_regionbox_1979-2014.nc'

    test_period = ('2005', '2014'),
    run_models: bool = False
    epoch_index: int = 50

    projection = False
    projection_path = None


@dataclass
class ProjectionConfig(Config):
    
    fname_gan_constrained = '/data/poem_gan_constraint_ssp585.nc'
    fname_gan_unconstrained = '/data/poem_gan_ssp585.nc'
    fname = '/data/poem_gan_constraint_ssp585.nc'
    fname_gfdl = '/data/cmip6_gfdl_ssp585.nc'
    fname_mpi = '/data/cmip6_mpi_ssp585.nc'
    fname_cesm = '/data/cmip6_cesm2_ssp585.nc'
    fname_poem = '/data/poem_ssp585.nc'

    test_period = ('2019', '2100')
    run_models: bool = False
    epoch_index: int = 50

    projection = True
    projection_path = fname_poem


