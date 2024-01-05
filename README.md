# PWP_self
My cluster (Ubuntu 22.0) version for PWP 1-d ocean model with some changes!

```
cd /big_drive/kaushik/PWP_runs/AS_65E_15N
```

import necessary Libraries, Half of these aren't even necessary

```
import matplotlib.pyplot as plt
import cosima_cookbook as cc
from tqdm import tqdm_notebook
import IPython.display
%matplotlib inline
import xarray as xr
import cartopy.crs as ccrs
import cartopy
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import numpy as np
import cmocean as cmo
import netCDF4
from netCDF4 import Dataset
import datetime as datetime
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from xoverturning import calcmoc
import gsw
```

Download Necessary Forcing files from ERA5

```
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_surface_latent_heat_flux',
            'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux', 'mean_surface_sensible_heat_flux',
            'mean_total_precipitation_rate',
        ],
        'year': '2020',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', 
            '03:00',
            '06:00', 
            '09:00',
            '12:00',
            '15:00',
            '18:00',
            '21:00',
        ],
        'area': [
            16, 63, 14,
            67,
        ],
    },
    '/big_drive/kaushik/PWP_runs/AS_65E_15N/era5_65e15n_2020.nc')
```

Initial Condition File Generation

```
# input_file = xr.open_dataset("ocean_daily_2006.nc")
input_file = xr.open_mfdataset('/big_drive/kaushik/Datasets/Observation_Products/ARGO/98*.nc')

# Depth as a dimension

z_lev = input_file.LEV
z_lev = z_lev[:27]
```

```
# Temp 

temp = input_file.PTEMP
temp = temp.isel(LEV=slice(0,27))
temp = temp.sel(LAT96_120=["15."], LON51_100=["65."], method='nearest')[0,:,0,0]
# t = t.isel(time=0, latitude=0, longitude=0)

# Salinity

salt = input_file.SALT 
# s = salt.sel(latitude=["15."], longitude=["65."], method='nearest')[0,:,:,:]
salt = salt.isel(LEV=slice(0,27))
salt = salt.sel(LAT91_120=["15."], LON41_80=["65."], method='nearest')[0,:,0,0]

lati = s.latitude
longi = s.longitude
```

```
ncfile = Dataset('/big_drive/kaushik/PWP_runs/AS_55E_15N/input_data/initial_condition_65e15n_2020.nc',mode='w',format='NETCDF4_CLASSIC')

lat_dim = ncfile.createDimension('lat',1)
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees'
lat.long_name = 'degrees latitude'

lon_dim = ncfile.createDimension('lon',1)
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees'
lon.long_name = 'degrees longitude'

z_dim = ncfile.createDimension('z',40)
z = ncfile.createVariable('z', np.float32, ('z',))
z.units = 'meters'
z.long_name = 'depth in meters'

t = ncfile.createVariable('t',np.float64,('z')) # note: unlimited dimension is leftmost
t.units = 'degrees C' # degrees Kelvin
t.standard_name = 'Ocean Temperature' # this is a CF standard name

s = ncfile.createVariable('s',np.float64,('z')) # note: unlimited dimension is leftmost
s.units = 'PSUs' # degrees Kelvin
s.standard_name = 'Ocean Salinity' # this is a CF standard name

nz = 1
lat[:] = lati
lon[:] = longi
z[:] = z_lev
t[:] = temp
s[:] = salt

print("-- Wrote data, t.shape is now ", t.shape)

ncfile.close()
```

Forcing File Tidying

```
forcings_file = xr.open_dataset("era5_65e15n_2020.nc")
forcings_file = forcings_file.sel(longitude=["65."], latitude=["15."], method="nearest")
forcings_file = forcings_file.isel(longitude=0, latitude=0)
```

State for how many days the model will be run (Here I have taken runtime=365days with \delta T=3hourly

```
# Creating a relative time axis
rtime = np.arange(0,365,0.125)
len(rtime)
```

Convert wind velocity to wind stress

```
rho_0 = 1.23
cd = 1.4e-3
u = forcings_file["u10"]
v = forcings_file["v10"]

U = np.sqrt(u*u + v*v)
tx = rho_0*cd*U*u 
ty = rho_0*cd*U*v
```

Convert rain from kg m-2 s-1 to m s-1 

```
ppt = forcings_file["mtpr"]*0.001
```

```
# Create dataset
forcings = xr.Dataset(
    data_vars=dict(
        qlat=(["time"], forcings_file.mslhf.values),
        tx=(["time"], tx.values),
        ty=(["time"], ty.values),
        sw=(["time"], forcings_file.msnswrf.values),
        lw=(["time"], forcings_file.msnlwrf.values),
        precip=(["time"], ppt.values),
        qsens=(["time"], forcings_file.msshf.values),
    ),
    coords=dict(
        dtime=(["dtime"], forcings_file.time.values),
        time=(["time"], rtime), 
    ),

    attrs=dict(description="Forcings for 55E 15N")
)
```

```
forcings.to_netcdf("forcings_65e15n_2020.nc")
```

Run the Model!

```
import PWP_helper as phf
import PWP
import seawater as sw
def exp1run1():
    forcing_fname = 'forcings_65e15n_2020.nc'
    prof_fname = 'initial_condition_65e15n_2020.nc'
    print("Running Case with data from Arabian Sea...")
    p={}
    p['rkz']=1e-6
    p['dz'] = 1.0
    p['max_depth'] = 500.0 
    p['rg'] = 0.25
    p['rb'] = 0.65
    p['lat'] = 15.
    p['winds_ON'] = True
    p['emp_ON'] = True
    p['heat_ON'] = True
    p['drag_ON'] = True 

    suffix = 'exp1_run1'
    forcing, pwp_out = PWP.run(met_data=forcing_fname, prof_data=prof_fname, suffix=suffix, save_plots=True, param_kwds=p)
```

```
exp1run1()
```










































