import yaml
import cdsapi
from datetime import datetime, timedelta
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
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep)+1

        self.grib_dir = self.cfg['output']['grib']
        self.netcdf_dir = self.cfg['output']['netcdf']
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)
        self.area_list = [
            self.cfg['area']['north'], 
            self.cfg['area']['west'], 
            self.cfg['area']['south'], 
            self.cfg['area']['east']
        ]
                self.client = cdsapi.Client()
        self.cdo = Cdo()
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
        build the request of CDS API

        Args:
            dataset (str): name of the dataset to reqest
            variables (list): list of the variables to download
            current_time (datetime): time of the dataset to reqest
            levels (list, optional): 壓力層級列表。默認為 None。

        Returns:
            dict: CDS API 請求字典。
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

    def download(self):
        """
        下載 ERA5 資料並轉換為 NetCDF 格式。
        """
        for i in tqdm(range(self.total_steps), desc="Downloading ERA5", unit="step"):
            current = self.start_t + i * self.a_timestep
            timestamp = current.strftime('%Y%m%d_%H%M')
            
            if "upper_dataset" in self.cfg:
                p = self.cfg["upper_dataset"]
                pl_grb = os.path.join(
                    f"{self.cfg['output']['grib']}", 
                    f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.grib"
                )
                pl_nc = os.path.join(
                    f"{self.cfg['output']['netcdf']}", 
                    f"{self.cfg['output']['prefix']['pres_lev']}_{timestamp}.nc"
                )
                req = self._build_request(
                    p['title'], 
                    p['variables'], 
                    current, 
                    p.get('levels')
                )
                self.client.retrieve(p['title'], req).download(pl_grb)
            if "surface_dataset" in self.cfg:
                s = self.cfg["surface_dataset"]
                sl_grb = os.path.join(
                    f"{self.cfg['output']['grib']}", 
                    f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.grib"
                )
                pl_nc = os.path.join(
                    f"{self.cfg['output']['netcdf']}", 
                    f"{self.cfg['output']['prefix']['sing_lev']}_{timestamp}.nc"
                )
                req = self._build_request(
                    s['title'], 
                    s['variables'], 
                    current,
                )
                self.client.retrieve(s['title'], req).download(sl_grb)


if __name__ == "__main__":
    config_file = "./era5.yaml"
    downloader = ERA5DataLoader(config_file)
    downloader.download()
