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
        return req

    def process_download(self, curr_time):
        self.pl = self.cfg["download"]["dataset_upper"]
        self.sl = self.cfg["download"]["dataset_surface"]
        timestamp = curr_time.strftime(self.timestr_fmt)

        final_nc = os.path.join(
            self.netcdf_dir,
            f"{self.prefix['output']}_{timestamp}.nc"
        )

        if os.path.exists(final_nc):
            print(f"[INFO] File exists, skipping: {final_nc}")
            return

        pl_grb = os.path.join(
            self.grib_dir,
            f"{self.prefix['upper']}_{timestamp}.grib"
        )
        sl_grb = os.path.join(
            self.grib_dir,
            f"{self.prefix['surface']}_{timestamp}.grib"
        )

        req_pl = self._build_request(self.pl['title'], self.pl['variables'], curr_time, self.pl.get('levels'))
        self.client.retrieve(self.pl['title'], req_pl).download(pl_grb)

        req_sl = self._build_request(self.sl['title'], self.sl['variables'], curr_time)
        self.client.retrieve(self.sl['title'], req_sl).download(sl_grb)

        # Chained CDO command: merge and then invert latitude
        self.cdo.invertlat(
            input=self.cdo.merge(input=f"{pl_grb} {sl_grb}", options="-f nc4 --eccodes"),
            output=final_nc,
        )

        # Clean up grib files
        os.remove(pl_grb)
        os.remove(sl_grb)

# if __name__ == "__main__":
#     config_file = "/wk2/yaochu/RESEARCH/dlamp/DLAMP.data.beta/config/era5.yaml"
#     downloader = CDSDataDownloader(config_file)
#     downloader.process_download()
