import yaml
import cdsapi
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm
import os

"""
share:
  time_control:

download:
  file_io:
  area:
  variables:

regrid:
diagnostics:
"""

class CDSDataDownloader:
    def __init__(self, yaml_path):
        self.cfg = self._load_config(yaml_path)

        self.cfg_time = self.cfg["share"]["time_control"]
        self.start_t = datetime.strptime(self.cfg_time["start"], self.cfg_time["format"])
        self.end_t = datetime.strptime(self.cfg_time["end"], self.cfg_time["format"])
        self.a_timestep = timedelta(hours=self.cfg_time["base_step_hours"])
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep) + 1

        self.cfg_io = self.cfg["share"]["file_io"]
        self.grib_dir = self.cfg_io['grib_dir']
        self.netcdf_dir = self.cfg_io['netcdf_dir']
        self.prefix = self.cfg_io["prefix"]
        self.pl_prefix = self.prefix["pres_lev"]
        self.sl_prefix = self.prefix["sing_lev"]
        self.timestr_fmt = self.prefix['timestr_fmt']
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)

        self.area_list = [
            self.cfg['area']['north'],
            self.cfg['area']['west'],
            self.cfg['area']['south'],
            self.cfg['area']['east']
        ]

        self.client = cdsapi.Client()

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

    def convert_grib_to_netcdf(self, grib_file: str, netcdf_file: str):
        """
        使用 xarray 讀取 GRIB 並轉為 NetCDF，並反轉緯度順序（南到北）
        """
        try:
            ds = xr.open_dataset(grib_file, engine="cfgrib")

            # 嘗試找到緯度維度並反轉
            for lat_name in ["latitude", "lat"]:
                if lat_name in ds.dims:
                    ds = ds.sortby(lat_name, ascending=True)
                    break

            ds.to_netcdf(netcdf_file, format="NETCDF4")
            ds.close()
        except Exception as e:
            print(f"[ERROR] GRIB 轉換失敗: {grib_file}\n{e}")

    def process_download(self):
        for i in tqdm(range(self.total_steps), desc="Downloading ERA5", unit="step"):
            curr_time = self.start_t + i * self.a_timestep
            timestamp = curr_time.strftime(self.timestr_fmt)

            pl_grb = os.path.join(self.grib_dir, f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.grib")
            pl_nc = os.path.join(self.netcdf_dir, f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.nc")
            sl_grb = os.path.join(self.grib_dir, f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.grib")
            sl_nc = os.path.join(self.netcdf_dir, f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.nc")

            # 壓力層資料
            if "dataset_upper" in self.cfg:
                p = self.cfg["dataset_upper"]
                req = self._build_request(p['title'], p['variables'], curr_time, p.get('levels'))
                self.client.retrieve(p['title'], req).download(pl_grb)
                self.convert_grib_to_netcdf(pl_grb, pl_nc)

            # 單層資料
            if "dataset_surface" in self.cfg:
                s = self.cfg["dataset_surface"]
                req = self._build_request(s['title'], s['variables'], curr_time)
                self.client.retrieve(s['title'], req).download(sl_grb)
                self.convert_grib_to_netcdf(sl_grb, sl_nc)


if __name__ == "__main__":
    config_file = "../config/era5.yaml"
    downloader = ERA5DataLoader(config_file)
    downloader.process_download()
