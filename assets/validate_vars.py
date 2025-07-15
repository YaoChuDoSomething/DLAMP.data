import os
import sys
os.system('clear')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from scipy.interpolate import griddata, RegularGridInterpolator

era5s_dataf = "../ncdb/Pool/era5sl_20220911_0000.nc"
rwrf_dataf = "/wk2/data/2020/20200520/0000/wrfinput_d01_2020-05-20_00_interp"
dlamp_dataf = "../ncdb/Pool/e5dlamp_20220911_0000.nc"


with xr.open_dataset(era5s_dataf, engine="netcdf4") as e5ds:
    lon = np.squeeze(e5ds["lon"].values)
    lat = np.squeeze(e5ds["lat"].values)
    
    elon, elat = np.meshgrid(lon, lat)
    e5sst = np.squeeze(e5ds["sst"].values)
    e5sst[np.isnan(e5sst)] = np.nanmean(e5sst.ravel())
    
    
    #land_mask = ~np.isnan(e5sst)
    #known_points = np.array([elon[land_mask], elat[land_mask]]).T
    #known_values = sst_data_with_nan[valid_mask]
    #mean_sst = np.nanmean(e5sst.ravel())


 

with xr.open_dataset(rwrf_dataf, engine="netcdf4") as rwfds:
    XLON = np.squeeze(rwfds["XLONG"].values)
    XLAT = np.squeeze(rwfds["XLAT"].values)
    SST_0 = np.squeeze(rwfds["SST"].values)
    T2_0 = np.squeeze(rwfds["T2"].values)
    dT_0 = T2_0 - SST_0

with xr.open_dataset(dlamp_dataf, engine="netcdf4") as ncds:
    XLON = np.squeeze(ncds["XLONG"].values)
    XLAT = np.squeeze(ncds["XLAT"].values)
    SST = np.squeeze(ncds["SST"].values)
    mask = np.isnan(SST) * 1.
    
    T2 = np.squeeze(ncds["T2"].values)
    dT = SST - T2
    
    

points = list(zip(elon.ravel(), elat.ravel()))

SST_1 = griddata(
    points, e5sst.ravel(), (XLON, XLAT), 
    method="linear",
)
SST_2 = griddata(
    points, e5sst.ravel(), (XLON, XLAT),
    method="cubic",
)
SST_3 = griddata(
    points, e5sst.ravel(), (XLON, XLAT), 
    method="linear", fill_value=0,
)


fig = plt.figure(figsize=[12, 8])
gs  = gridspec.GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax_dict = {
    'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 
    'ax4': ax4, 'ax5': ax5, 'ax6': ax6,
}

img1 = ax1.pcolormesh(SST_0)
ax1.set_title("SST")
img2 = ax2.pcolormesh(T2_0)
ax2.set_title("T2")
img3 = ax3.pcolormesh(dT_0)
ax3.set_title("SST - T2")
img4 = ax4.pcolormesh(e5sst)
ax4.set_title("era5 SST")
img5 = ax5.pcolormesh(SST_3)
ax5.set_title("era5 SST with mean SST Mask")
#img6 = ax6.pcolormesh(e5sst_n)
#ax6.set_title("era5 new SST")

