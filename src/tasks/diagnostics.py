from src.core.pipeline import Task
from src.core.context import Context
from src.registry.diagnostic_registry import (
    load_diagnostics,
    sort_diagnostics_by_dependencies,
)
import xarray as xr
import os
from datetime import datetime, timedelta


class Diagnostics(Task):
    def execute(self, context: Context):
        print("[INFO] Starting Diagnostics...")
        self.cfg = context.config

        # Load Registry
        self.diagnostics = load_diagnostics(context.config_path)
        self.source_dataset = self.cfg["registry"]["source_dataset"]

        netcdf_dir = self.cfg["share"]["io_control"]["netcdf_dir"]
        prefix = self.cfg["share"]["io_control"]["prefix"]
        timestr_fmt = prefix["timestr_fmt"]

        time_cfg = self.cfg["share"]["time_control"]
        start_t = datetime.strptime(time_cfg["start"], time_cfg["format"])
        end_t = datetime.strptime(time_cfg["end"], time_cfg["format"])
        out_step = timedelta(hours=self.cfg["regrid"]["output_step_hours"])
        total_steps = int((end_t - start_t) / out_step) + 1

        for i in range(total_steps):
            curr_time = start_t + out_step * i
            timestamp = curr_time.strftime(timestr_fmt)

            # Input: Try Regridded file first (E.g. e5regrid_...), else Raw (e5dlamp_...)
            regrid_nc = os.path.join(netcdf_dir, f"{prefix['regrid']}_{timestamp}.nc")
            input_nc = os.path.join(
                netcdf_dir, f"{prefix['output']}_{timestamp}.nc"
            )  # 'raw' converted

            final_nc = os.path.join(
                netcdf_dir, f"{prefix['output']}_diag_{timestamp}.nc"
            )

            target_file = None
            if os.path.exists(regrid_nc):
                target_file = regrid_nc
            elif os.path.exists(input_nc):
                # Fallback to raw if regridding wasn't done/files missing
                # But assuming this pipeline aims for regridded output usually.
                target_file = input_nc
            else:
                # print(f"[WARN] No input file for diagnostics at {timestamp}")
                continue

            if os.path.exists(final_nc):
                # print(f"[INFO] Diagnostic file exists: {final_nc}")
                continue

            print(f"[DIAGNOSE] Processing {target_file}")

            # Load Dataset
            # Using context manager to ensure close
            with xr.open_dataset(target_file) as ds:
                # We need to perform calculations.
                # Since we modify the dataset, we might want to load it fully or copy it to avoid read-only issues with some engines,
                # but usually 'r' mode + adding variables in memory works if we write to NEW file.
                # Actually xarray (netcdf4) might need explicit load to modify comfortably?
                # Let's load it into memory.
                outds = ds.load()

                ordered_vars = sort_diagnostics_by_dependencies(self.diagnostics)
                for var in ordered_vars:
                    if var not in self.diagnostics:
                        continue

                    info = self.diagnostics[var]
                    requires = info["requires"]
                    diag_func = info["function"]

                    if all(
                        req in outds.data_vars or req in outds.coords
                        for req in requires
                    ):
                        try:
                            # Passing source_dataset string (e.g. "ERA5") and the dataset object
                            data_arr = diag_func(self.source_dataset, outds)
                            outds[var] = data_arr
                            print(f"  > Calculated {var}, shape: {data_arr.shape}")
                        except Exception as e:
                            print(f"  > [ERROR] Failed {var}: {e}")
                    else:
                        pass
                        # missing = [
                        #     r
                        #     for r in requires
                        #     if r not in outds.data_vars and r not in outds.coords
                        # ]
                        # print(f"  > [SKIP] {var} missing dependencies: {missing}")

                # Save Result
                outds.to_netcdf(final_nc, format="NETCDF4")
                print(f"[DONE] Saved diagnostic file: {final_nc}")
