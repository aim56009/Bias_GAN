import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from Bias_GAN.code.src.data import TestData


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
                map_resolution='c',
                vmin_contours=0.15,
                vmax_contours=0.40,
                mask_threshold=1):

    import matplotlib.pyplot as plt
    
    if axis is not None:
        cbar_plt = plt
        plt = axis

    lats = data.lat
    lons = data.lon

    if projection == 'mill':
        lon_0 = 0
    else:
        lon_0 = -180 
    
    m = Basemap(llcrnrlon=lons[0], llcrnrlat=lats[0],
                urcrnrlon=lons[-1], urcrnrlat=lats[-1],
                projection=projection, lon_0=lon_0, 
                resolution=map_resolution, ax=axis)
    
    m.drawcounties()
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


class PlotAnalysis():
    
    def __init__(self, data: TestData):
        
        self.data = data
        self.names = ['era5', 'gan', 'climate_model']
                 

    def single_frames(self, vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False,
                              projection="robin"):
        fig, axs = plt.subplots(1, 3, figsize=(12,7),  constrained_layout=True)
        alpha = 1.0 
        name = ['era5','gan', 'climate_model']
        letters = ['a', 'b', 'c']

        for ax, (name, letter) in zip(axs, zip(name, letters)):
            data = abs(getattr(self.data, name).isel(time=time_index))
            if name == 'era5':
                print(data.time.values)

            ax.annotate(f"{letter}", ha="center", va="center", size=15,
                        xy=(1-0.955, 0.925), xycoords=ax,
                        bbox=None) 
            ax.set_title(self.data.model_name_definition(name))
            plt.title(self.data.model_name_definition(name))

            cbar = False
            cbar_title = ''

            cs = plot_basemap(data, cbar_title, vmin, vmax, alpha, cmap,
                        cbar=cbar,
                        axis=ax,
                        return_cs=True,
                        projection=projection,
                        map_resolution='c',
                        plot_mask=mask)
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs,
                     location='bottom',
                     shrink=0.4,
                     aspect=10,
                     extend='max'
                     ).set_label('Precipitation [mm/d]', fontsize=13)


    def histograms(self, single_plot=False, ax=None, show_legend=True, annotate=True,log=True,xlim_end=400):

        if single_plot:
            plt.figure(figsize=(6,4))

        if ax is None:
            _, ax = plt.subplots()

        for name in reversed(self.names):

            if name == 'era5':
                alpha = 1.0
            else:
                alpha = 0.9

            data = getattr(self.data, name).values.flatten()
            era5_data = getattr(self.data, "era5").values.flatten()
            if name != "era5":
              print(f" Absolute Difference era5&{name} (in test period):",int(np.sum(era5_data)-np.sum(data)))
              print('')
              print(f" Absolute Difference in percent era5&{name}",abs(np.round((np.sum(era5_data)-np.sum(data))/np.sum(era5_data),5)))
              print('')
              counts, bin_edges = np.histogram(data, density=False, bins=100)
              bin_widths = np.diff(bin_edges)
              area = np.sum(counts * bin_widths)
              print(f"Area under the Histogram era5&{name}",int(np.round(area)))
              if name == "climate_model":
                print("_______________________________________________________________")
                print('')



            
            label = self.data.model_name_definition(name)
            _ = plt.hist(data,
                        bins=100,
                        histtype='step',
                        log=log,
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
        #plt.xlim(0,xlim_end)
        plt.grid()
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

            plt.legend(handles=reversed(new_handles), labels=reversed(labels))

        if single_plot:
            plt.show()

    def latitudinal_mean(self, single_plot=False, ax=None, show_legend=True, annotate=True):
        if single_plot:
            plt.figure(figsize=(12,8))
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
            if name != "era5":
                era5_data = getattr(self.data, "era5")
                bias = data.mean('time') - era5_data.mean('time') 
                print(label,f" \t \t MAE: {abs(bias).values.mean():2.3f} [K]")

            data = data.mean(dim=("lon", "time"))
            
            plt.plot(data.lat, data,
                     label=label,
                     alpha=alpha,
                     linestyle=linestyle,
                     linewidth=2,
                     color=self.data.colors(name))
        
        if ax is not None and annotate is True:
           ax.annotate("a", ha="center", va="center", size=15,
                     xy=(1-0.955, 0.925), xycoords=ax,
                     bbox=None) 

        plt.xlim(25,58)
        plt.xlabel('Latitude')
        plt.ylabel('Mean temperature [K]')
        plt.grid()

        if show_legend:
            plt.legend(loc='upper right')

        if single_plot:
            plt.show()       


    def avg_frames_abs_err(self, vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False,
                              projection="robin",
                              scale_precip_by=1):
        fig, axs = plt.subplots(1, 2, figsize=(12,7),  constrained_layout=True)
        alpha = 1.0 
        name = ['gan', 'climate_model']
        letters = ['a', 'b']

        for ax, (name, letter) in zip(axs, zip(name, letters)):

            era5_data = np.mean(abs(getattr(self.data, 'era5')),axis=0) 
            data = np.mean(abs(getattr(self.data, name)),axis=0)

            ax.annotate(f"{letter}", ha="center", va="center", size=15,
                        xy=(1-0.955, 0.925), xycoords=ax,
                        bbox=None) 
            ax.set_title(self.data.model_name_definition(name))
            plt.title(self.data.model_name_definition(name))

            cbar_title = ''
            cs = plot_basemap(scale_precip_by*(era5_data-data), cbar_title, vmin, vmax, alpha, cmap,
                        cbar=False,
                        axis=ax,
                        return_cs=True,
                        projection=projection,
                        map_resolution='c',
                        plot_mask=mask)
            print(f"abs error era5_data- {name}:", np.round(np.sum(np.abs(era5_data-data)).values,2))
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs,
                     location='bottom',
                     shrink=0.4,
                     aspect=10,
                     extend='max'
                     ).set_label(f'Precipitation Difference to ERA5 [mm/d] x {scale_precip_by}', fontsize=13) 
        if scale_precip_by != 1:
            print(f"carefull: precipitation in the plot is scaled by a factor of {scale_precip_by}")


    def avg_frames(self, vmin=0,
                              vmax=20,
                              time_index=-1,
                              cmap='Blues',
                              mask=False,
                              single_plot=False,
                              projection="robin",
                              scale_precip_by=1):
        fig, axs = plt.subplots(1, 3, figsize=(12,7),  constrained_layout=True)
        alpha = 1.0 
        name = ['era5','gan', 'climate_model']
        letters = ['a', 'b', 'c']
        
        
        for ax, (name, letter) in zip(axs, zip(name, letters)):
            data = np.mean(abs(getattr(self.data, name)),axis=0)


            ax.annotate(f"{letter}", ha="center", va="center", size=15,
                        xy=(1-0.955, 0.925), xycoords=ax,
                        bbox=None) 
            ax.set_title(self.data.model_name_definition(name))
            plt.title(self.data.model_name_definition(name))

            cbar = False
            cbar_title = ''

            cs = plot_basemap(scale_precip_by*data, cbar_title, vmin, vmax, alpha, cmap,
                        cbar=cbar,
                        axis=ax,
                        return_cs=True,
                        projection=projection,
                        map_resolution='c',
                        plot_mask=mask)
                 
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,
                     ax=axs,
                     location='bottom',
                     shrink=0.4,
                     aspect=10,
                     extend='max'
                     ).set_label(f'Precipitation [mm/d] x {scale_precip_by}', fontsize=13)

        if scale_precip_by != 1:
            print(f"carefull: precipitation in the plot is scaled by a factor of {scale_precip_by}")



    def log_histograms(self, single_plot=False, ax=None, show_legend=True, annotate=True):

        if single_plot:
            plt.figure(figsize=(6,4))

        if ax is None:
            _, ax = plt.subplots()

        for name in reversed(self.names):

            if name == 'era5':
                alpha = 1.0
            else:
                alpha = 0.9

            
            era5_data = getattr(self.data, "era5").values.flatten() 
            data = getattr(self.data, name).values.flatten() 
            
            epsilon = 1e-10
            if name != "era5":
              print(f"Absolute Difference era5&{name} (in test period):",int(np.sum(np.log(era5_data+epsilon))-np.sum(np.log(data+epsilon))))
              print('')
              print(f"Absolute Difference in percent era5&{name}",abs(np.round((np.sum(np.log(era5_data+epsilon))-np.sum(np.log(data+epsilon)))/np.sum(abs(np.log(era5_data+epsilon))),3)))
              print('')
              counts, bin_edges = np.histogram(np.log(data+epsilon), bins=100)
              bin_widths = np.diff(bin_edges)
              area = np.sum(counts * bin_widths)
              print(f"Area under the Histogram era5&{name}",int(np.round(area)))
              if name == "climate_model":
                print("_______________________________________________________________")
                print('')

              

            label = self.data.model_name_definition(name)
            _ = plt.hist( np.log(data+epsilon),
                        bins=100,
                        histtype='step',
                        log=False,
                        label=label,
                        alpha=alpha,
                        density=True,
                        linewidth=2,
                        color=self.data.colors(name))
            
        
        if ax is not None and annotate is True:
            ax.annotate("b", ha="center", va="center", size=15,xy=(1-0.955, 0.925), xycoords=ax,bbox=None) 

        plt.xlabel(' Precipitation [mm/d]')
        plt.ylabel(' Precipitation Histogram')
        #plt.xlim(0,7)
        #plt.ylim(0,0.08)
        plt.grid()
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

            plt.legend(handles=reversed(new_handles), labels=reversed(labels))

        if single_plot:
            plt.show()

    def histogram_diff(self, single_plot=False, ax=None, show_legend=True, annotate=True,log=True,xlim_end=500):

        if single_plot:
            plt.figure(figsize=(6,4))

        if ax is None:
            _, ax = plt.subplots()

        for name in reversed(self.names):

            if name == 'era5':
                alpha = 1.0
                pass
            else:
                alpha = 0.9

            data = getattr(self.data, name).values.flatten()
            era5_data = getattr(self.data, "era5").values.flatten()
            if name != "era5":
              print(f"Absolute Difference era5&{name} (in test period):",int(np.sum(era5_data)-np.sum(data)))
              print('')
              print(f"Absolute Difference in percent era5&{name}", abs(np.round((np.sum(era5_data)-np.sum(data) )/np.sum(era5_data),5)) )
              print('')
              counts, bin_edges = np.histogram(abs(era5_data-data), bins=100)
              bin_widths = np.diff(bin_edges)
              area = np.sum(counts * bin_widths)
              print(f"Area under the Histogram era5&{name}",int(np.round(area)))
              if name == "climate_model":
                print("_______________________________________________________________")
                print('')

            label = self.data.model_name_definition(name)
            _ = plt.hist(abs(era5_data-data),
                        bins=100,
                        histtype='step',
                        log=log,
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
        plt.xlim(0,xlim_end)
        plt.grid()
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

            plt.legend(handles=reversed(new_handles), labels=reversed(labels))

        if single_plot:
            plt.show()


    def log_histogram_diff(self, single_plot=False, ax=None, show_legend=True, annotate=True,log=True):

        if single_plot:
            plt.figure(figsize=(6,4))

        if ax is None:
            _, ax = plt.subplots()

        for name in reversed(self.names):

            if name == 'era5':
                alpha = 1.0
                pass
            else:
                alpha = 0.9

            data = getattr(self.data, name).values.flatten()
            era5_data = getattr(self.data, "era5").values.flatten()

            epsilon = 1e-10
            if name != "era5":
              print(f"Absolute Difference era5&{name} (in test period):",int(np.sum(np.log(era5_data+epsilon))-np.sum(np.log(data+epsilon)) ))
              print('')
              print(f"Absolute Difference in percent era5&{name}",abs(np.round( (np.sum(np.log(era5_data+epsilon))- np.sum(np.log(data+epsilon)))/np.sum(np.log(era5_data+epsilon)),5)))
              print('')
              counts, bin_edges = np.histogram(abs(np.log(era5_data+epsilon)-np.log(data+epsilon)), bins=100)
              bin_widths = np.diff(bin_edges)
              area = np.sum(counts * bin_widths)
              print(f"Area under the Histogram era5&{name}",int(np.round(area)))
              if name == "climate_model":
                print("_______________________________________________________________")
                print('')
          
            label = self.data.model_name_definition(name)
            _ = plt.hist(abs(np.log(era5_data+epsilon)-np.log(data+epsilon)),
                        bins=100,
                        histtype='step',
                        log=log,
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
        plt.xlim(0,30)
        plt.grid()

        

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

            plt.legend(handles=reversed(new_handles), labels=reversed(labels))

        if single_plot:
            plt.show()

"""
def plot_reconstruction(plot_data, vmin=0,
                          vmax=20,
                          time_index=-1,
                          cmap='Blues',
                          mask=False,
                          single_plot=False,
                          projection="robin",
                          scale_precip_by=1):
    fig, axs = plt.subplots(1, 1, figsize=(12,7),  constrained_layout=True)
    alpha = 1.0 
    data = abs(plot_data["gan_reconstruct"].isel(time=time_index))
  
    plt.title("reconstructed gan data ")
    cbar = False
    cbar_title = ''
    cs = plot_basemap(scale_precip_by*data, cbar_title, vmin, vmax, alpha, cmap,cbar=cbar,axis=None,return_cs=True,projection=projection,map_resolution='c',plot_mask=mask)
              
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax=axs,location='bottom',shrink=0.4,aspect=10,extend='max').set_label('Precipitation [mm/d]', fontsize=13)
    
def plot_reconstruction_day(plot_data, vmin=0,
                          vmax=20,
                          time_index=-1,
                          cmap='Blues',
                          mask=False,
                          single_plot=False,
                          projection="robin",
                          scale_precip_by=1):
    fig, axs = plt.subplots(1, 1, figsize=(12,7),  constrained_layout=True)
    alpha = 1.0 
    data = abs(plot_data)
  
    plt.title("reconstructed gan data ")
    cbar = False
    cbar_title = ''
    cs = plot_basemap(scale_precip_by*data, cbar_title, vmin, vmax, alpha, cmap,cbar=cbar,axis=None,return_cs=True,projection=projection,map_resolution='c',plot_mask=mask)
              
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax=axs,location='bottom',shrink=0.4,aspect=10,extend='max').set_label('Precipitation [mm/d]', fontsize=13)
    
def plot_gan(plot_data, vmin=0,
                          vmax=20,
                          time_index=-1,
                          cmap='Blues',
                          mask=False,
                          single_plot=False,
                          projection="robin",
                          scale_precip_by=1):
    fig, axs = plt.subplots(1, 1, figsize=(12,7),  constrained_layout=True)
    alpha = 1.0 
    data = plot_data
  
    plt.title("reconstructed gan data ")
    cbar = False
    cbar_title = ''
    cs = plot_basemap(scale_precip_by*data, cbar_title, vmin, vmax, alpha, cmap,cbar=cbar,axis=None,return_cs=True,projection=projection,map_resolution='c',plot_mask=mask)
              
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax=axs,location='bottom',shrink=0.4,aspect=10,extend='max').set_label('Precipitation [mm/d]', fontsize=13)
"""
