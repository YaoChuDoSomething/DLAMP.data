import yaml
import cdsapi
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from cdo import Cdo
import os

class ERA5DataLoader:
    def __init__(self, yaml_path):
        ### ##### #####
        # YAML config 
        ##### ##### ###
        self.cfg = self._load_config(yaml_path)

        ### ##### #####
        # time control 
        ##### ##### ###
        self.start_t = datetime.strptime(
            self.cfg['setup_time']['start'],
            self.cfg['setup_time']['format']
        )
        self.end_t = datetime.strptime(
            self.cfg['setup_time']['end'], 
            self.cfg['setup_time']['format']
        )
        self.a_timestep = timedelta(
            hours=self.cfg['setup_time']['base_step_hours']
        )
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep)+1

        ### ##### #####
        # I/O control
        ##### ##### ###
        self.grib_dir = self.cfg['output']['grib']
        self.netcdf_dir = self.cfg['output']['netcdf']
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)
        
        ### ##### #####
        # Geo-domain range control 
        ##### ##### ###
        self.area_list = [
            self.cfg['area']['north'], 
            self.cfg['area']['west'], 
            self.cfg['area']['south'], 
            self.cfg['area']['east']
        ]
        
        ### ##### #####
        # CDS API Client setup 
        ##### ##### ###
        self.client = cdsapi.Client()
        
        ### ##### #####
        # Initialization of cdo modules
        ##### ##### ###
        self.cdo = Cdo(tempdir="./cdo_tmp")
        self.cdo.debug = True

    def _load_config(self, yaml_path):
        """
        import YAML format file for configuration

        Returns:
            dict: content of configuration
        """
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def _build_request(self, dataset, variables, current, levels=None):
        """
        Builds the request dictionary for the CDS (Copernicus Climate Change Service) API.

        Args:
            dataset (str): Name of the dataset to request.
            variables (list): List of the variables to download.
            current_time (datetime): Time of the dataset to request.
            levels (list, optional): List of pressure levels. Defaults to None.

        Returns:
            dict: CDS API request dictionary.
        """
        req = {
            "product_type": "reanalysis",
            "year": current.strftime('%Y'),
            "month": current.strftime('%m'),
            "day": current.strftime('%d'),
            "time": current.strftime('%H:%M'),
            "variable": variables,
            "format": "grib"
        }
        if levels:
            req["pressure_level"] = [str(l) for l in levels]
        if self.area_list:
            req["area"] = self.area_list
        return req
    
    def invert_grids_in_lat(input_file: str, output_file: str):
        """
        Inverts the xarray Dataset along the latitude dimension and its associated datasets.

        Args:
            input_file (str): Path to the input NetCDF file.
            output_file (str): Path to the output NetCDF file.
        """
        with xr.open_dataset(input_file, engine="netcdf4") as ds:
            lat_dim_name = None
            common_lat_names = 'lat'
            for dim in common_lat_names:
                if dim in ds.dims:
                    lat_dim_name = dim
                    break
            
            ds[lat_dim_name] = ds[lat_dim_name][::-1]
            for var_name, da in ds.data_vars.items():
                if lat_dim_name in da.dims:
                    ds[var_name] = da.isel({lat_dim_name: slice(None, None, -1)})
                else:
                    print("Var: ", var_name, " has no latitudes component")
            ds.to_netcdf(output_file, format='NETCDF4')
            

    def process_download(self):
        """
        下載 ERA5 資料並轉換為 NetCDF 格式。
        """
        for i in tqdm(range(self.total_steps), desc="Downloading ERA5", unit="step"):
            current = self.start_t + i * self.a_timestep
            timestamp = current.strftime('%Y%m%d_%H%M')
            pl_grb = os.path.join(
                f"{self.cfg['output']['grib']}", 
                f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.grib"
            )
            pl_nc = os.path.join(
                f"{self.cfg['output']['netcdf']}", 
                f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.nc"
            )
            sl_grb = os.path.join(
                f"{self.cfg['output']['grib']}", 
                f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.grib"
            )
            sl_nc = os.path.join(
                f"{self.cfg['output']['netcdf']}", 
                f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.nc"
            )
            
            
            if "dataset_upper" in self.cfg:
                p = self.cfg["dataset_upper"]
                req = self._build_request(
                    p['title'], 
                    p['variables'], 
                    current, 
                    p.get('levels')
                )
                self.client.retrieve(p['title'], req).download(pl_grb)
                self.cdo.invertlat(
                    input=self.cdo.copy(
                        input=pl_grb,
                        option="-f nc4 --eccodes",
                    ),
                    option="-f nc4",
                    output=pl_nc
                )
                
                
            if "dataset_surface" in self.cfg:
                s = self.cfg["dataset_surface"]
                req = self._build_request(
                    s['title'], 
                    s['variables'], 
                    current,
                )
                self.client.retrieve(s['title'], req).download(sl_grb)
                self.cdo.invertlat(
                    input=self.cdo.copy(
                        input=sl_grb,
                        option="-f nc4 --eccodes",
                    ),
                    option="-f nc4",
                    output=sl_nc
                )


#if __name__ == "__main__":
#    config_file = "./era5.yaml"
#    downloader = ERA5DataLoader(config_file)
#    downloader.download()
