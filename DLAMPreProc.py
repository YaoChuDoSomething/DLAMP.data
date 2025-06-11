import yaml
import numpy as np
import xarray as xr
from src.cds_downloader import ERA5DataLoader

## ERA5 Dataset Downloading ###
era5_config = "./config/era5.yaml"
downloader = ERA5DataLoader(era5_config)
downloader.process_download()

### ###

