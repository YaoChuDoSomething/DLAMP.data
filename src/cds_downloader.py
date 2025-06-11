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
        self.cfg = self._load_config(yaml_path)

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
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep) + 1

        self.grib_dir = self.cfg['output']['grib']
        self.netcdf_dir = self.cfg['output']['netcdf']
        self.timestr_fmt = self.cfg['output']['prefix']['timestr_fmt']
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)

        self.area_list = [
            self.cfg['area']['north'],
            self.cfg['area']['west'],
            self.cfg['area']['south'],
            self.cfg['area']['east']
        ]

        self.client = cdsapi.Client()
        self.cdo = Cdo(tempdir="./cdo_tmp")
        self.cdo.debug = True

    def _load_config(self, yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def _build_request(self, dataset, variables, current_time, levels=None):
        req = {
            "product_type": "reanalysis",
            "year": current_time.strftime('%Y'),
            "month": current_time.strftime('%m'),
            "day": current_time.strftime('%d'),
            "time": [current_time.strftime('%H:%M')],
            "variable": variables,
            "format": "grib"
        }
        if levels:
            req["pressure_level"] = [str(l) for l in levels]
        if self.area_list:
            req["area"] = self.area_list
        return req

    def invert_grids_in_lat(self, input_file: str, output_file: str):
        with xr.open_dataset(input_file, engine="netcdf4") as ds:
            lat_dim_name = None
            for dim in ['latitude', 'lat', 'y']:
                if dim in ds.dims:
                    lat_dim_name = dim
                    break

            if lat_dim_name is None:
                raise ValueError("No latitude dimension found.")

            ds[lat_dim_name] = ds[lat_dim_name][::-1]
            for var_name, da in ds.data_vars.items():
                if lat_dim_name in da.dims:
                    ds[var_name] = da.isel({lat_dim_name: slice(None, None, -1)})
            ds.to_netcdf(output_file, format='NETCDF4')

    def process_download(self):
        for i in tqdm(range(self.total_steps), desc="Downloading ERA5", unit="step"):
            curr_time = self.start_t + i * self.a_timestep
            timestamp = curr_time.strftime(self.timestr_fmt)

            pl_grb = os.path.join(self.grib_dir, f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.grib")
            pl_nc = os.path.join(self.netcdf_dir, f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.nc")
            sl_grb = os.path.join(self.grib_dir, f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.grib")
            sl_nc = os.path.join(self.netcdf_dir, f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.nc")

            if "dataset_upper" in self.cfg:
                p = self.cfg["dataset_upper"]
                req = self._build_request(p['title'], p['variables'], curr_time, p.get('levels'))
                self.client.retrieve(p['title'], req).download(pl_grb)
                self.cdo.invertlat(
                    input=self.cdo.copy(input=pl_grb, options="-f nc4 --eccodes"),
                    options="-f nc4",
                    output=pl_nc
                )

            if "dataset_surface" in self.cfg:
                s = self.cfg["dataset_surface"]
                req = self._build_request(s['title'], s['variables'], curr_time)
                self.client.retrieve(s['title'], req).download(sl_grb)
                self.cdo.invertlat(
                    input=self.cdo.copy(input=sl_grb, options="-f nc4 --eccodes"),
                    options="-f nc4",
                    output=sl_nc
                )

if __name__ == "__main__":
    config_file = "../config/era5.yaml"
    downloader = ERA5DataLoader(config_file)
    downloader.process_download()
