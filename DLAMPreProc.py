#!/bin/python

DLAMP_DATA_DIR = "./"

###===== ERA5 Dataset Downloading ===================================###
#   Downloading process is fully controled by YAML configure file
###==================================================================###

from src.cds_downloader import CDSDataDownloader

era5_config = f"{DLAMP_DATA_DIR}/config/era5.yaml"
downloader = CDSDataDownloader(era5_config)
downloader.process_download()


###===== DataRegridder and Variables Registry =======================###
#   Data Regridding can be controled by YAML configure file
#   Variables Registry can be controled by YAML configure file
#   Variables Diagnostics can be controled by module script:
#       src/registry/diagnostics_functions
###==================================================================###

from src.diag_variables import DataRegridder

yaml_file = f"{DLAMP_DATA_DIR}/config/DLAMPreproc.yaml"
regridder = DataRegridder(yaml_file)
regridder.main_process()


