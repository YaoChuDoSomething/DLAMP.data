#!/bin/python

#!/bin/python

###===== Workflow Control ===========================================###
#
###==================================================================###

do_cds_downloader = 1
do_dlamp_regridder = 1

DLAMP_DATA_DIR = "./"


###===== ERA5 Dataset Downloading ===================================###
#   Downloading process is fully controled by YAML configure file
###==================================================================###



if do_cds_downloader == 1:
    from src.preproc.cds_downloader import CDSDataDownloader

    era5_config = f"{DLAMP_DATA_DIR}/config/era5.yaml"
    downloader = CDSDataDownloader(era5_config)
    timeline = downloader.create_timeline()
    
    for curr in timeline:
       downloader.process_download(curr)


###===== DataRegridder and Variables Registry =======================###
#   Data Regridding can be controled by YAML configure file
#   Variables Registry can be controled by YAML configure file
#   Variables Diagnostics can be controled by module script:
#       src/registry/diagnostics_functions
###==================================================================###

if do_dlamp_regridder == 1:
    from src.preproc.dlamp_regridder import DataRegridder

    #yaml_file = f"{DLAMP_DATA_DIR}/config/era5.yaml"
    regridder = DataRegridder(era5_config)
    regridder.process_single_time(curr)
    
    #regridder.main_process()


