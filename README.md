# PWP_self
My cluster (Ubuntu 22.0) version for PWP 1-d ocean model with some changes!

```
cd /mnt/12tb_hdd/WORK/PWP_test
## Make sure this folder has the .py functions PWP.py and PWP_helper.py
```

import necessary Libraries, Half of these aren't even necessary

```
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import netCDF4
from netCDF4 import Dataset
```

Download Necessary Forcing files from ERA5

```
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_surface_latent_heat_flux",
        "mean_surface_net_long_wave_radiation_flux",
        "mean_surface_net_short_wave_radiation_flux",
        "mean_surface_sensible_heat_flux",
        "mean_total_precipitation_rate"
    ],
    "year": ["2020"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [16, 63, 14, 67]  ### region of interest [lat_max, lon_min, lat_min, lon_max]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```
Prepare the ERA data
```
!cdo -f nc copy 8bb8de8cae5e59a6beebe58863205ee1.grib forcing.nc   #### convert the grib file to an nc file
!cdo chname,\10u,u10 forcing.nc out.nc; mv out.nc forcing.nc       #### rename u10 and v10 variable names
!cdo chname,\10v,v10 forcing.nc out.nc; mv out.nc forcing.nc
```

Initial Condition File Generation

```
# Temp initialised from WOA23 dataset

temp = xr.open_dataset('/mnt/12tb_hdd/WORK/PWP_test/woa23_B5C2_t01_04.nc',decode_times=False).t_an
temp = temp.sel(lat=15, lon=65, method='nearest')[0,:]

# Salinity initialised from WOA23 dataset

salt = xr.open_dataset('/mnt/12tb_hdd/WORK/PWP_test/woa23_B5C2_s01_04.nc',decode_times=False).s_an 
salt = salt.sel(lat=15, lon=65, method='nearest')[0,:]

# lati = salt.LAT91_120
# longi = salt.LON41_80

lati = 15
longi = 65

z_lev = temp.depth
z_lev = z_lev[:998]
```

```
# Temp 

temp = input_file.TEMP
temp = temp[0,:].isel(DEPTH=slice(0,998))
# temp = temp.sel(LAT96_120=["15."], LON51_100=["65."], method='nearest')[:,:,0,0].sel(TIME=('2020-06'))
# t = t.isel(time=0, latitude=0, longitude=0)

# Salinity

salt = input_file.PSAL 
# s = salt.sel(latitude=["15."], longitude=["65."], method='nearest')[0,:,:,:]
salt = salt[0,:].isel(DEPTH=slice(0,998))
# salt = salt.sel(LAT91_120=["15."], LON41_80=["65."], method='nearest')[:,:,0,0].sel(TIME=('2020-06'))

# lati = salt.LAT91_120
# longi = salt.LON41_80

lati = 13.86
longi = 67.64
```

```
ncfile = Dataset('/mnt/12tb_hdd/WORK/PWP_test/initial_condition_65e15n_2020.nc',mode='w',format='NETCDF4_CLASSIC')

lat_dim = ncfile.createDimension('lat',1)
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees'
lat.long_name = 'degrees latitude'

lon_dim = ncfile.createDimension('lon',1)
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees'
lon.long_name = 'degrees longitude'

z_dim = ncfile.createDimension('z',57)
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
forcings_file = xr.open_dataset("/mnt/12tb_hdd/WORK/PWP_test/forcing.nc")
forcings_file = forcings_file.sel(lon=["65."], lat=["15."], method="nearest")
forcings_file = forcings_file.isel(lon=0, lat=0)
```

State for how many days the model will be run (Here I have taken runtime=365days with \delta T=3hourly

```
# Creating a relative time axis
rtime = np.arange(0,366,0.125)
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
forcings.to_netcdf("/mnt/12tb_hdd/WORK/PWP_test/forcings_65e15n_2020.nc")
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

Make some necessary folders to tidy your run
```
!mkdir output
!mkdir plots
!mkdir input_data
!mv forcings_65e15n_2020.nc initial_condition_65e15n_2020.nc input_data
```


```
exp1run1()
```










































