import yaml
import cdsapi
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from cdo import Cdo
import os


class CDSDataDownloader:
    def __init__(self, yaml_path):
        self.cfg = self._load_config(yaml_path)

        self.cfg_time = self.cfg["share"]["time_control"]
        self.start_t = datetime.strptime(
            self.cfg_time["start"], 
            self.cfg_time["format"]
        )
        self.end_t = datetime.strptime(
            self.cfg_time["end"], 
            self.cfg_time["format"]
        )
        self.a_timestep = timedelta(
            hours=self.cfg_time["base_step_hours"]
        )
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep) + 1

        self.io = self.cfg["share"]["io_control"]
        self.base_dir = self.io["base_dir"]
        self.grib_dir = os.path.join(self.base_dir, self.io["grib_subdir"])
        self.netcdf_dir = os.path.join(self.base_dir, self.io["netcdf_subdir"])
        self.prefix = self.io["prefix"]
        self.timestr_fmt = self.prefix['timestr_fmt']
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)

        self.area = self.cfg["download"]["area"]
        self.area_list = [
            self.area['north'],
            self.area['west'],
            self.area['south'],
            self.area['east'],
        ]

        self.client = cdsapi.Client()
        self.cdo = Cdo(tempdir="./.cdo_tmp")
        self.cdo.debug = True

    def _load_config(self, yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def create_timeline(self):
        return [
            self.start_t + t * self.a_timestep
            for t in range(self.total_steps)
        ]

    def _build_request(self, dataset, variables, curr_time, levels=None):
        req = {
            "product_type": "reanalysis",
            "year": [curr_time.strftime('%Y')],
            "month": [curr_time.strftime('%m')],
            "day": [curr_time.strftime('%d')],
            "time": [curr_time.strftime('%H:%M')],
            "variable": variables,
            "format": "grib"
        }
        if levels:
            req["pressure_level"] = [str(l) for l in levels]
        if self.area_list:
            req["area"] = self.area_list
        return req

    def invertlat_to_netcdf(self, input_grib: str, output_netcdf: str):
        """
        Using xarray to read grib data, and invert latitude and the data depend on, then
        save the data in netcdf format.
        """
        try:
            ds = xr.open_dataset(input_grib, engine="cfgrib")
            for lat_name in ["latitude", "lat"]:
                if lat_name in ds.dims:
                    ds = ds.sortby(lat_name, ascending=True)
                    break

            ds.to_netcdf(output_netcdf, format="netcdf4")
            ds.close()
        except Exception as e:
            print(f"[ERROR] GRIB failed converting: {input_grib}\n{e}")

    def process_download(self, curr_time):
        self.pl = self.cfg["download"]["dataset_upper"]
        self.sl = self.cfg["download"]["dataset_surface"]
        #for i in tqdm(range(self.total_steps), desc="Downloading ERA5", unit="step"):
            #curr_time = self.start_t + i * self.a_timestep
        timestamp = curr_time.strftime(self.timestr_fmt)

        pl_grb = os.path.join(
            self.grib_dir, 
            f"{self.prefix['upper']}_{timestamp}.grib"
        )
        pl_nc = os.path.join(
            self.netcdf_dir, 
            f"{self.prefix['upper']}_{timestamp}.nc"
        )
        sl_grb = os.path.join(
            self.grib_dir, 
            f"{self.prefix['surface']}_{timestamp}.grib"
        )
        sl_nc = os.path.join(
            self.netcdf_dir,
            f"{self.prefix['surface']}_{timestamp}.nc"
        )

        if not os.path.exists(pl_grb):
            req = self._build_request(self.pl['title'], self.pl['variables'], curr_time, self.pl.get('levels'))
            self.client.retrieve(self.pl['title'], req).download(pl_grb)
        if not os.path.exists(pl_nc):
            #self.invertlat_to_netcdf(input_grib=pl_grb, output_netcdf=pl_nc)
            self.cdo.invertlat(
                input=pl_grb,
                options="-f nc4 --eccodes",
                output=pl_nc,
            )
        if not os.path.exists(sl_grb):
            req = self._build_request(self.sl['title'], self.sl['variables'], curr_time)
            self.client.retrieve(self.sl['title'], req).download(sl_grb)
        if not os.path.exists(sl_nc):
            #self.invertlat_to_netcdf(input_grib=sl_grb, output_netcdf=sl_nc)
            self.cdo.invertlat(
                input=sl_grb,
                options="-f nc4 --eccodes",
                output=sl_nc,
            )

# if __name__ == "__main__":
#     config_file = "/wk2/yaochu/RESEARCH/dlamp/DLAMP.data.beta/config/era5.yaml"
#     downloader = CDSDataDownloader(config_file)
#     downloader.process_download()
