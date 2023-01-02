import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from Bias_GAN.code.src.data import TestData



class PlotAnalysis():
    
    def __init__(self, data: TestData):
        
        self.data = data

        self.names = ['era5', 'poem', 'gan']
                 #self.names = ['era5', 'poem', 'cmip_model',   \
                 #'quantile_mapping', 'gan_constrained']

    def single_frames(self, 
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False):
        fig, axs = plt.subplots(2,2,figsize=(12,7),  constrained_layout=True)
        alpha = 1.0 

        name = ['era5',"era5",'gan', 'poem' ]

        letters = ['a', 'c', 'b',"d"]

        i = 0
        for col in range(2):
            for row in range(2):

                data = abs(getattr(self.data, name[i]).isel(time=time_index))
                if i == 0: print(data.time.values)

                ax = axs[row, col]
                ax = plt.subplot(ax)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
                plt.title(self.data.model_name_definition(name[i]))

                cbar = False
                cbar_title = ''

                cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=cbar,
                             axis=ax,
                             return_cs=True,
                             projection='robin',
                             map_resolution='c',
                             plot_mask=mask)
                i += 1
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs[1,:],
                     location='bottom',
                     shrink=0.4,
                     aspect=10,
                     extend='max'
                     ).set_label('Precipitation [mm/d]', fontsize=13)


    def single_frames_with_spectrum(self, spectral_density,
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              fontsize=None,
                              single_plot=False):

        fig = plt.figure(constrained_layout=True, figsize=(14, 17))
        subfigs = fig.subfigures(2, 1, wspace=0.1, hspace=0.025)        
        alpha = 1.0 

        name = ['era5', 'poem', 'gan',"gan"]

        letters = ['a', 'b', 'c',"gan"]

        axsLeft = subfigs[0].subplots(2, 2, sharey=True)
        i = 0
        for ax in axsLeft.flatten():

            data = abs(getattr(self.data, name[i]).isel(time=time_index))
            if i == 0: print(data.time.values)

            ax.annotate(f"{letters[i]}", ha="center", va="center", size=20,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 
            ax.set_title(self.data.model_name_definition(name[i]), fontsize=fontsize)

            cbar = False
            cbar_title = ''

            cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                         cbar=cbar,
                         axis=ax,
                         return_cs=True,
                         projection='robin',
                         map_resolution='c',
                         plot_mask=mask)
            i += 1
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])
        subfigs[0].colorbar(sm,
                            shrink=0.3,
                            aspect=25,
                            ax=axsLeft,
                            extend='max',
                            location='bottom').set_label('Precipitation [mm/d]', 
                            fontsize=fontsize)


        axsRight = subfigs[1].subplots(1, 1, sharex=True)
        axsRight.annotate("e", ha="center", va="center", size=20,
                 xy=(0.955, 0.925), xycoords=axsRight,
                 bbox=None) 
        spectral_density.plot(axis=axsRight, fontsize=fontsize, linewidth=2.2)

    def single_frames_with_spectrum_seprate_figure(self, spectral_density,
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              fontsize=None,
                              single_plot=False):

        fig = plt.figure(constrained_layout=True, figsize=(14, 17))       
        alpha = 1.0 

        name = ['era5', 'poem', 'gan']

        letters = ['a', 'b', 'c',"gan"]

        axsLeft = subfigs[0].subplots(2, 2, sharey=True)
        i = 0
        for ax in axsLeft.flatten():

            data = abs(getattr(self.data, name[i]).isel(time=time_index))
            if i == 0: print(data.time.values)

            ax.annotate(f"{letters[i]}", ha="center", va="center", size=20,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 
            ax.set_title(self.data.model_name_definition(name[i]), fontsize=fontsize)

            cbar = False
            cbar_title = ''

            cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                         cbar=cbar,
                         axis=ax,
                         return_cs=True,
                         projection='robin',
                         map_resolution='c',
                         plot_mask=mask)
            i += 1
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])
        subfigs[0].colorbar(sm,
                            shrink=0.3,
                            aspect=25,
                            ax=axsLeft,
                            extend='max',
                            location='bottom').set_label('Precipitation [mm/d]', 
                            fontsize=fontsize)


        axsRight = subfigs[1].subplots(1, 1, sharex=True)
        axsRight.annotate("e", ha="center", va="center", size=20,
                 xy=(0.955, 0.925), xycoords=axsRight,
                 bbox=None) 
        spectral_density.plot(axis=axsRight, fontsize=fontsize, linewidth=2.2)


    def single_frames_cmip(self, 
                              vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False):
        fig, axs = plt.subplots(3,2,figsize=(12,10.3),  constrained_layout=True)
        alpha = 1.0 

        name = ['era5', 'poem',  'gan_constrained', \
                 'gfdl', 'mpi', 'cesm2']

        letters = ['a', 'b', 'c', 'd', 'e', 'f']

        i = 0
        for row in range(3):
            for col in range(2):

                data = abs(getattr(self.data, name[i]).isel(time=time_index))
                if i == 0: print(data.time.values)

                ax = axs[row, col]
                ax = plt.subplot(ax)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                         xy=(1-0.955, 0.925), xycoords=ax,
                         bbox=None) 
                plt.title(self.data.model_name_definition(name[i]))

                cbar = False
                cbar_title = ''

                cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=cbar,
                             axis=ax,
                             return_cs=True,
                             projection='robin',
                             map_resolution='c',
                             plot_mask=mask)
                i += 1
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs[2,:],
                     location='bottom',
                     shrink=0.4,
                     aspect=10,
                     extend='max'
                     ).set_label('Precipitation [mm/d]', fontsize=13)

    def era5_poem_comparison(self,
                             vmin=0,
                             vmax=20,
                             cmap='Blues',
                             time_index=-1,
                             single_plot=False):

        if single_plot:
            plt.figure(figsize=(15,8))

        alpha = 0.7

        data = abs(self.data.era5).isel(time=time_index)
        plt.subplot(2,1,1)
        plt.title(f"ERA5")
        cbar_title = 'Precipitation [mm/d]'
        plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap, cbar=True, plot_mask=True)

        data = abs(self.data.poem).isel(time=time_index)
        plt.subplot(2,1,2)
        plt.title(f"POEM")
        cbar_title = 'Precipitation [mm/d]'
        plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap, cbar=True, plot_mask=True)

        if single_plot:
            plt.show()

        
    def histograms(self, single_plot=False, ax=None, show_legend=True, annotate=True):

        if single_plot:
            plt.figure(figsize=(6,4))

        for name in reversed(self.names):

            if name == 'era5':
                alpha = 1.0
            else:
                alpha = 0.9

            data = getattr(self.data, name).values.flatten()
            label = self.data.model_name_definition(name)
            _ = plt.hist(data,
                         bins=100,
                         histtype='step',
                         log=True,
                         label=label,
                         alpha=alpha,
                         density=True,
                         linewidth=2,
                         color=self.data.colors(name))

        if ax is not None and annotate is True:
           ax.annotate("b", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 
        
        plt.xlabel('Precipitation [mm/d]')
        plt.ylabel('Histogram')
        plt.xlim(0,360)
        plt.grid()
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

            plt.legend(handles=reversed(new_handles), labels=reversed(labels))

        if single_plot:
            plt.show()

        
    def latitudinal_mean(self, single_plot=False, ax=None, show_legend=True, annotate=True):
        
        if single_plot:
            plt.figure(figsize=(6,4))
        names = self.names
        for name in reversed(names):
            if name == 'era5':
                alpha = 1.0
                linestyle = '--'
            else:
                alpha = 0.9
                linestyle= '-'

            data = getattr(self.data, name)
            label = self.data.model_name_definition(name)
            data = data.mean(dim=("longitude", "time"))
            plt.plot(data.latitude, data,
                     label=label,
                     alpha=alpha,
                     linestyle=linestyle,
                     linewidth=2,
                     color=self.data.colors(name))
        
        if ax is not None and annotate is True:
           ax.annotate("a", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 

        plt.ylim(0,7.5)
        plt.xlim(-80,80)
        plt.xlabel('Latitude')
        plt.ylabel('Mean precipitation [mm/d]')
        plt.grid()

        if show_legend:
            plt.legend(loc='upper right')

        if single_plot:
            plt.show()
        

    def latitudinal_mean_and_histograms(self, single_plot=False):
        if single_plot:
            plt.figure(figsize=(14,4))

        ax = plt.subplot(1,2,1)
        self.latitudinal_mean(single_plot=single_plot, ax=ax,
                                          show_legend=False )

        ax = plt.subplot(1,2,2)
        self.histograms(single_plot=single_plot, ax=ax)

        if single_plot:
            plt.show()


    def bias_all(self,
                      vmin=-10,
                      vmax=10,
                      cmap="seismic",
                      single_plot=False,
                      season=None):

        #plt.figure(figsize=(14, 15))
        fig, axs = plt.subplots(3,2,figsize=(13, 16),  constrained_layout=True)

        alpha = 0.6
        letters = ['a', 'b', 'c', 'd', 'e', 'f']
        
        # excluding era5 form self.names
        era5 = self.data.era5
        count = 1

        for row in range(3):
            for col in range(2):

                ax = axs[row, col]
                ax = plt.subplot(ax)

                ax.annotate(f"{letters[count-1]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                if count == 5:
                    self.latitudinal_mean(single_plot=False,
                                          ax=ax,
                                          show_legend=False, 
                                          annotate=False)

                elif count == 6:
                    self.histograms(single_plot=False,
                                    ax=ax,
                                    annotate=False)

                else:
                    name = self.names[count]
                    if name != 'gan':
                    
                        data = getattr(self.data, name)
                        if season is not None:
                            data = data.where(data['time'].dt.season==season, drop=True)
                            era5 = era5.where(era5['time'].dt.season==season, drop=True)
                        bias = data.mean('time') - era5.mean('time') 

                    label = self.data.model_name_definition(name)
                    print(label,f" \t \t MAE: {abs(bias).values.mean():2.3f} [mm/d]")
                    plt.title(label)


                    cbar_title = f'ME [mm/d]'
                    if count == 1:
                        parallel_label = [1,0,0,0]
                        meridian_label = [0,0,0,1]
                    elif count == 2:
                        parallel_label = [1,0,0,0]
                        meridian_label = [0,0,0,1]
                    elif count == 3:
                        parallel_label = [1,0,0,0]
                        meridian_label = [0,0,0,1]
                    elif count == 4:
                        parallel_label = [1,0,0,0]
                        meridian_label = [0,0,0,1]

                    if count == 3 or count == 5:
                        plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                     cbar=False,
                                     cbar_extend='both',
                                     parallel_label=parallel_label,
                                     meridian_label=meridian_label,
                                     draw_coordinates=True, axis=ax)
                    else:
                        plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                     cbar=False, cbar_extend='both',
                                     parallel_label=parallel_label,
                                     meridian_label=meridian_label,
                                     draw_coordinates=True, axis=ax)
                count += 1

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs[1,:],
                     location='bottom',
                     shrink=0.3,
                     aspect=12,
                     extend='both'
                     ).set_label('ME [mm/d]', fontsize=13)


    def bias(self,
                      vmin=-10,
                      vmax=10,
                      cmap="seismic",
                      single_plot=False,
                      season=None):

        
        plt.figure(figsize=(12.5,8.5))

        alpha = 0.6

        letters = ['a', 'b', 'c', 'd']
        
        # excluding era5 form self.names
        era5 = self.data.era5
        count = 1
        for i,name in enumerate(self.names[1:]):
            
            if name != 'gan':

                data = getattr(self.data, name)
                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)
                    era5 = era5.where(era5['time'].dt.season==season, drop=True)
                bias = data.mean('time') - era5.mean('time') 

                ax = plt.subplot(2,2,count)
                count += 1
                label = self.data.model_name_definition(name)
                print(label,f" \t \t MAE: {abs(bias).values.mean():2.3f} [mm/d]")
                plt.title(label)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                cbar_title = f'ME [mm/d]'
                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]

                if i == 0  or i == 2:
                    plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=False,
                                 cbar_extend='both',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                 draw_coordinates=True, axis=ax)
                else:
                    plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=True, cbar_extend='both',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                 draw_coordinates=True, axis=ax)


    def extremes_bias(self,
                      quantile,
                      vmin=-10,
                      vmax=10,
                      cmap="seismic",
                      single_plot=False,
                      season=None):

        
        plt.figure(figsize=(12.5,8.5))

        alpha = 0.6

        letters = ['a', 'b', 'c', 'd']
        
        # excluding era5 form self.names
        era5 = self.data.era5
        count = 1
        min_precipitation_threshold = 0.5
        for i,name in enumerate(self.names[1:]):
            
            if name != 'gan':

                data = getattr(self.data, name)
                if season is not None:
                    data = data.where(data['time'].dt.season==season, drop=True)
                    era5 = era5.where(era5['time'].dt.season==season, drop=True)

                bias = data.where(data > min_precipitation_threshold, drop=True).quantile(quantile, dim='time', skipna=True) - \
                       era5.where(era5 > min_precipitation_threshold, drop=True).quantile(quantile, dim='time', skipna=True)

                ax = plt.subplot(2,2,count)
                count += 1
                label = self.data.model_name_definition(name)
                print(label,f" \t \t MAE: {abs(bias).mean(skipna=True).values:2.3f} [mm/d]")
                plt.title(label)

                ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

                cbar_title = f'ME [mm/d]'
                if i == 0:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 1:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,0]
                elif i == 2:
                    parallel_label = [1,0,0,0]
                    meridian_label = [0,0,0,1]
                elif i == 3:
                    parallel_label = [0,0,0,0]
                    meridian_label = [0,0,0,1]

                if i == 0  or i == 2:
                    plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=False,
                                 cbar_extend='both',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                 draw_coordinates=True, axis=ax)
                else:
                    plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                                 cbar=True, cbar_extend='both',
                                 parallel_label=parallel_label,
                                 meridian_label=meridian_label,
                                 draw_coordinates=True, axis=ax)


    def bias_cmip(self,
                      vmin=-10,
                      vmax=10,
                      cmap="seismic",
                      single_plot=False,
                      season=None):

        
        plt.figure(figsize=(12.5,8.5))
        alpha = 0.6

        letters = ['a', 'b', 'c', 'd']
        names = ['poem', 'gfdl', 'mpi', 'cesm2']
        
        era5 = self.data.era5
        count = 1
        for i,name in enumerate(names[:]):
            data = getattr(self.data, name)
            if season is not None:
                data = data.where(data['time'].dt.season==season, drop=True)
                era5 = era5.where(era5['time'].dt.season==season, drop=True)
            bias = data.mean('time') - era5.mean('time') 

            ax = plt.subplot(2,2,count)
            count += 1

            label = self.data.model_name_definition(name)
            print(label,f" \t \t MAE: {abs(bias).values.mean():2.3f} [mm/d]")
            plt.title(label)

            ax.annotate(f"{letters[i]}", ha="center", va="center", size=15,
                 xy=(1-0.955, 0.925), xycoords=ax,
                 bbox=dict(boxstyle="square, pad=0.25", fc="white", ec="k", lw=1)) 

            cbar_title = f'ME [mm/d]'
            if i == 0:
                parallel_label = [1,0,0,0]
                meridian_label = [0,0,0,0]
            elif i == 1:
                parallel_label = [0,0,0,0]
                meridian_label = [0,0,0,0]
            elif i == 2:
                parallel_label = [1,0,0,0]
                meridian_label = [0,0,0,1]
            elif i == 3:
                parallel_label = [0,0,0,0]
                meridian_label = [0,0,0,1]

            if i == 0  or i == 2:
                plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=False,
                             cbar_extend='both',
                             parallel_label=parallel_label,
                             meridian_label=meridian_label,
                             draw_coordinates=True, axis=ax)
            else:
                plot_basemap(bias, cbar_title, vmin, vmax, alpha, cmap,
                             cbar=True, cbar_extend='both',
                             parallel_label=parallel_label,
                             meridian_label=meridian_label,
                             draw_coordinates=True, axis=ax)     


    def power_spectrum(self, num_pixel=60):

        fig1, ax1 = plt.subplots()

        data = self.data.era5
        mean_bins, kvals = compute_mean_spectrum(data)
        kvals_degrees = 1/kvals*num_pixel
        ax1.plot(kvals_degrees, mean_bins, label='ERA5')

        data = self.data.gan
        mean_bins, kvals = compute_mean_spectrum(data)
        ax1.plot(kvals_degrees, mean_bins, label='GAN')

        data = self.data.climate_model
        mean_bins, kvals = compute_mean_spectrum(data)
        ax1.plot(kvals_degrees, mean_bins, label='POEM')

        ax1.set_xscale('log')
        ax1.set_xlabel('k [deg]')
        ax1.set_ylabel('P(k)')
        ax1.set_xticks([2, 2.5, 5, 10, 20, 40, 60])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend()
        plt.grid()

        plt.show()


def plot_basemap(data: xr.DataArray,
                cbar_title: str,
                vmin: float,
                vmax: float,
                alpha: float,
                cmap: str,
                cbar=True,
                cbar_extend='max',
                cbar_position='right',
                return_cs=False,
                axis=None,
                plot_mask=False,
                draw_coordinates=False,
                parallel_label=[1, 0, 0, 0],
                meridian_label=[0, 0, 0, 1],
                contours=None,
                fig=None,
                projection='mill',
                contourf=False,
                map_resolution='l',
                vmin_contours=0.15,
                vmax_contours=0.40,
                mask_threshold=1):

    import matplotlib.pyplot as plt
    
    if axis is not None:
        cbar_plt = plt
        plt = axis

    lats = data.latitude
    lons = data.longitude

    if projection == 'mill':
        lon_0 = 0
    else:
        lon_0 = -180 

    m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                projection=projection, lon_0=lon_0, 
                resolution=map_resolution, ax=axis)

    m.drawcoastlines(color='k', linewidth=0.5)
                    
    if draw_coordinates:
        par = m.drawparallels(
                              [-90, -60, 0, 60, 90],
                              #[-90, -45, 0, 45, 90],
                              linewidth=1.0,
                              labels=parallel_label,
                              color='grey')

        merid = m.drawmeridians(
                                [ -90, 0, 90, 180],
                                #[-120, -60, 0, 60, 120, 180],
                                linewidth=1.0,
                                labels=meridian_label,
                                color='grey')
    
    Lon, Lat = np.meshgrid(lons, lats)
    
    x, y = m(Lon, Lat)
                    
    if contourf:
        cs = plt.contourf(x, y, data, 500, vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap,
                        linewidth=0, shading='auto', extend='max')
    else:
        cs = plt.pcolormesh(x, y, data, vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap,
                        linewidth=0, shading='auto')

    if plot_mask is True:
        mask = np.ma.masked_where(data > mask_threshold, data)
        plt.pcolormesh(x,y, mask, vmin=-1, vmax=-1, alpha=1.0, cmap='Greys',shading='auto')

    if contours is not None:
        cs2 = plt.contour(x, y, abs(contours), 8, 
                            alpha=1.0, cmap='YlOrRd',
                            linewidth=4.0, shading='auto')
    if axis is None:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_position, size="1.5%", pad=0.4)
        cbar_plt = cax
    if cbar:

        if axis is not None:
            ax = cbar_plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_position, size="2.5%", pad=0.2)
            cbar = cbar_plt.colorbar(cs, cax=cax,  label=cbar_title, extend=cbar_extend)
        else:
            cbar = plt.colorbar(cs, cax=cax,  label=cbar_title, extend=cbar_extend)

        cbar.solids.set(alpha=1)
        if cbar_position == 'left':
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')

        if contours is not None:
            norm = matplotlib.colors.Normalize(vmin=cs2.cvalues.min(), vmax=cs2.cvalues.max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap = cs2.cmap)
            sm.set_array([])

            #cax = divider.new_vertical(size="3%", pad=0.2, pack_start=True)
            cax = divider.append_axes('right', size="1.5%", pad=0.2)
            #fig.add_axes(cax)
            fig.colorbar(sm,
                         ticks=cs2.levels,
                         cax=cax,
                         #orientation="horizontal",
                         orientation="vertical",
                         extend='max',
                         label='Feature importance [a.u.]')

    else:
        #cax = divider.append_axes("right", size="2%", pad=0.15)
        if axis is None:
            cax.set_visible(False)
    if return_cs:
        return cs


def plot_single_projection(data,
                           title='',
                           ylim=None,
                           y_label=True,
                           x_label=True,
                           x_ticks=True,
                           ax=None,
                          markevery=20):
    x_axis = data.gan.time.values
    plt.plot(x_axis, data.gan.values, label='GAN (unconstrained)', color=data.colors('gan'))
    plt.plot(x_axis, data.gan_constrained.values, label='GAN', color=data.colors('gan_constrained'))
    plt.plot(x_axis, data.poem.values, 'x', markersize=5, markeredgewidth=1.5, markevery=markevery, label='CM2Mc-LPJmL', color=data.colors('poem'))
    plt.plot(x_axis, data.cmip_model.values, label='GFDL-ESM4', color=data.colors('cmip_model'))
    if x_label is True:
        plt.xlabel('Year')
    if y_label is True:
        plt.ylabel('Averaged precipitation [mm/d]')
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.grid(True)
    if x_ticks is False:
        ax.xaxis.set_ticklabels([]) 
        ax.xaxis.set_ticks_position('none')


def plot_projection(fname, projection_global, projection_tropics, projection_temperate, single_legend=False, markevery=20):

    #fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,16), sharex=True)
    plt.figure(figsize=(8,14))
    plt.rcParams.update({'font.size': 12})
    ax= plt.subplot(311)
    data = projection_global
    plot_single_projection(data,
                           title='Global',
                           ylim=(2.25,2.6),
                           x_label=False,
                           x_ticks=False,
                           ax=ax,
                           markevery=markevery)
    ax.annotate("a", ha="center", va="center", size=15,
                         xy=(1-0.965, 0.915), xycoords=ax,
                         bbox=None)

    plt.legend(bbox_to_anchor=(0.5, 0.6))

    ax = plt.subplot(312)
    data = projection_tropics
    plot_single_projection(data,
                           title='Tropics',
                           y_label=True,
                           x_label=False,
                           x_ticks=False,
                           ax=ax,
                           markevery=markevery)
    ax.annotate("b", ha="center", va="center", size=15,
                         xy=(1-0.965, 0.915), xycoords=ax,
                         bbox=None)

    if not single_legend: plt.legend()

    ax = plt.subplot(313)
    data = projection_temperate
    plot_single_projection(data,
                           title='Temperate zone',
                           ylim=(2.5,3.2),
                           markevery=markevery)
    ax.annotate("c", ha="center", va="center", size=15,
                         xy=(1-0.965, 0.915), xycoords=ax,
                         bbox=None)

    if not single_legend: plt.legend(bbox_to_anchor=(0.5, 0.6))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.15)

    plt.savefig(fname, format='pdf', bbox_inches='tight')
    print(fname)


def plot_single_projection_cmip(data,
                                title='',
                                ylim=None,
                                y_label=True,
                                x_label=True,
                                x_ticks=True,
                                ax=None):

    x_axis = data.poem.time.values
    plt.plot(x_axis, data.poem.values, label='CM2Mc-LPJmL', color=data.colors('poem'))
    plt.plot(x_axis, data.gfdl.values,
             label=data.model_name_definition('gfdl'),
             color=data.colors('gfdl'))
    plt.plot(x_axis, data.mpi.values,
             label=data.model_name_definition('mpi'),
             color=data.colors('mpi'))
    plt.plot(x_axis, data.cesm2.values,
             label=data.model_name_definition('cesm2'),
             color=data.colors('cesm2'))
    if x_label is True:
        plt.xlabel('Year')
    if y_label is True:
        plt.ylabel('Averaged precipitation [mm/d]')
    if ylim is not None:
        plt.ylim(ylim)
    if x_ticks is False:
        ax.xaxis.set_ticklabels([]) 
        ax.xaxis.set_ticks_position('none')
    plt.title(title)
    plt.grid()
    

def plot_projection_cmip(fname, projection_global, projection_tropics, projection_temperate):

    plt.figure(figsize=(8,14))

    plt.rcParams.update({'font.size': 12})
    ax= plt.subplot(311)
    data = projection_global
    plot_single_projection_cmip(data, title='Global', ylim=(2.25,2.6),
                                x_label=False, x_ticks=False, ax=ax)
    ax.annotate("a", ha="center", va="center", size=15,
                         xy=(1-0.965, 0.915), xycoords=ax,
                         bbox=None)
    plt.legend(bbox_to_anchor=(0.5, 0.6))

    ax = plt.subplot(312)
    data = projection_tropics
    plot_single_projection_cmip(data, title='Tropics', y_label=True,
                                x_label=False, x_ticks=False, ax=ax)
    ax.annotate("b", ha="center", va="center", size=15,
                         xy=(1-0.965, 0.915), xycoords=ax,
                         bbox=None)
    plt.legend()

    ax = plt.subplot(313)
    data = projection_temperate
    plot_single_projection_cmip(data, title='Temperate zone', ylim=(2.5,3.2))
    ax.annotate("c", ha="center", va="center", size=15,
                xy=(1-0.965, 0.915), xycoords=ax,
                bbox=None)
    plt.legend(bbox_to_anchor=(0.5, 0.6))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.1, hspace=0.15)

    plt.savefig(fname, format='pdf', bbox_inches='tight')
    print(fname)

def plot_gan_ensemble():
    test_split = ('2001', '2014')
    fig_fname = f'/results/gan_ensemble_lat_mean_histogram.pdf'

    ensemble_fname = '/data/gan_ensemble.nc'
    ds = xr.open_dataset(ensemble_fname).sel(time=slice(test_split[0], test_split[1]))

    era5_fname = '/data/era5.nc'
    era5 = xr.open_dataset(era5_fname).sel(time=slice(test_split[0], test_split[1]))*3600*24

    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(15,5))

    ax = plt.subplot(1,2,1)
    ax.annotate("a", ha="center", va="center", size=15,
                 xy=(1-0.955, 0.925), xycoords=ax,
                 bbox=None) 

    data = ds.gan_ensemble.mean(dim=("longitude","member", "time"))
    ax.plot(data.latitude, data,
            label='GAN ensemble mean',
            alpha=1,
            #linestyle=None,
            linewidth=2,
            color='grey')
    std = ds.gan_ensemble.mean(dim=("longitude", "time")).std(dim='member')
    ax.fill_between(data.latitude, data-std, data+std ,alpha=0.3, facecolor='grey')

    data = era5.era5_precipitation.mean(dim=("longitude", "time"))
    ax.plot(data.latitude, data,
            label='ERA5',
            alpha=1,
            linestyle='--',
            linewidth=2,
            color='k')

    plt.ylim(0,7.5)
    plt.xlim(-80,80)
    plt.xlabel('Latitude')
    plt.ylabel('Mean precipitation [mm/d]')
    plt.grid()

    ax = plt.subplot(1,2,2)
    data = era5.era5_precipitation.values.flatten()
    val, bin = np.histogram(data,
                     bins=100,
                     #histtype='step',
                     #log=True,
                     density=True,
                     #linewidth=2,
                     )

    ax.plot(bin[:-1], val, '--', label='ERA5', color='k')

    vals = []
    bins = []
    for i in range(10):
        data = ds.gan_ensemble.isel(member=i).values.flatten()
        val, bin = np.histogram(data,
                         bins=100,
                         #histtype='step',
                         #log=True,
                         density=True,
                         #linewidth=2,
                         )
        vals.append(val)
        bins.append(bin)

    bins = np.array(bins)
    vals = np.array(vals)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    ax.plot(bins[0][:-1], mean, color='grey', label='GAN ensemble mean')
    ax.fill_between(bins[0][:-1], mean-std, mean+std ,alpha=0.3, facecolor='grey')
    ax.set_yscale('log')

    ax.annotate("b", ha="center", va="center", size=15,
              xy=(1-0.955, 0.925), xycoords=ax,
              bbox=None) 
        
    plt.xlabel('Precipitation [mm/d]')
    plt.ylabel('Histogram')
    plt.xlim(0,320)
    plt.ylim(1e-9,2e-1)
    plt.grid()
    plt.legend()

    plt.savefig(fig_fname, format='pdf', bbox_inches='tight')
    print(fig_fname)
    plt.show()
