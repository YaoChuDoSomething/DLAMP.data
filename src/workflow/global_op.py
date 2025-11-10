#! src/workflow/global_op.py
from datetime import datetime, timedelta
import logging
import yaml

import xarray as xr
import numpy as np

from earth2studio.data import GFS
from earth2studio.models.px import SFNO
from earth2studio.io import NetCDF4Backend

log = logging.getLogger(__name__)

class GlobalOps:
    def __init__(self, cfg_path: str):
        with open("./config/sfno.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)

        sfno_pkg = SFNO.load_default_package()
        self.sfno_model = SFNO.load_model(sfno_pkg)

        self.datasource = GFS()
        self.io_backend = NetCDF4Backend(f"{self.cfg['io']['dir']}/{self.cfg['io']['prefix']}.nc")

    def one_way_run(self, start_time: datetime, lead_time: int):
        log.info(f"Starting one-way run from {start_time} with lead time {lead_time}")
        # Implement the one-way run logic here
        pass

    def two_way_run(self, start_time: datetime, lead_time: int, assimilation_interval: int):
        log.info(f"Starting two-way run from {start_time} with lead time {lead_time} and assimilation interval {assimilation_interval}")
        # Implement the two-way run logic here
        pass    

