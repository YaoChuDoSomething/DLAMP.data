import os
from datetime import datetime, timedelta
from src.core.pipeline import Task
from src.core.context import Context
from cdo import Cdo

cdo = Cdo()


class BaseConverter(Task):
    """Abstract base class for format conversion."""

    pass


class GribToNetCDF(BaseConverter):
    def execute(self, context: Context):
        print("[INFO] Starting Grib to NetCDF Conversion...")
        cfg = context.config
        cdo = Cdo(tempdir="./.cdo_tmp")
        cdo.debug = True

        # Re-derive timeline (this duplication suggests Context should hold the timeline iterator)
        # For this step, let's just repeat the logic or assume we process what's in GRIB dir?
        # Better: Repeat timeline logic for consistency.
        time_cfg = cfg["share"]["time_control"]
        start_t = datetime.strptime(time_cfg["start"], time_cfg["format"])
        end_t = datetime.strptime(time_cfg["end"], time_cfg["format"])
        step = timedelta(hours=time_cfg["base_step_hours"])
        total_steps = int((end_t - start_t) / step) + 1

        grib_dir = cfg["share"]["io_control"]["grib_dir"]
        netcdf_dir = cfg["share"]["io_control"]["netcdf_dir"]
        os.makedirs(netcdf_dir, exist_ok=True)

        prefix = cfg["share"]["io_control"]["prefix"]
        timestr_fmt = prefix["timestr_fmt"]

        for i in range(total_steps):
            curr_time = start_t + step * i
            timestamp = curr_time.strftime(timestr_fmt)

            combined_grb = os.path.join(
                grib_dir, f"{prefix['combined']}_{timestamp}.grib"
            )
            final_nc = os.path.join(netcdf_dir, f"{prefix['output']}_{timestamp}.nc")

            if os.path.exists(final_nc):
                # print(f"[INFO] NetCDF exists: {final_nc}")
                continue

            if os.path.exists(combined_grb):
                # CDO Invert Lat on the combined file
                print(f"[CONVERT] {timestamp}: Merged GRIB -> NetCDF4")
                try:
                    cdo.invertlat(
                        input=f"-f nc4 --eccodes {combined_grb}",
                        output=final_nc,
                    )
                except Exception as e:
                    print(f"[ERROR] CDO conversion failed for {timestamp}: {e}")
            else:
                print(f"[WARN] Missing merged GRIB file for {timestamp}")


class HsdToNetCDF(BaseConverter):
    """Reserved for Himawari HSD to NetCDF conversion."""

    def execute(self, context: Context):
        print("[INFO] HSD to NetCDF Reserved.")
