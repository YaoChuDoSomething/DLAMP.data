import yaml
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from src.registry.diagnostic_registry import load_diagnostics, sort_diagnostics_by_dependencies

from datetime import datetime, timedelta
import os

class DataRegridder:
    """
    DataRegridder
    │
    ├── __init__(...)
    ├── build_timeline(...)           # flexible time-control
    ├── process_single_time(...)      # basis: time
    ├── read_netcdf_data(...)         # basis: var
    ├── horizontal_interp(...)        # basis: source_var
    ├── diag_*()                      # basis: target_var
    │
    │
    │
    ├── write_output(...)             # basis: dict
    │
    └── main_process()                # A hot pot put everything-together
    """
    def __init__(self, yaml_path):
        # load YAML configure
        self.cfg = self._load_config(yaml_path)

        # time control
        self.cfg_time = self.cfg["share"]["time_control"]
        self.start_t = datetime.strptime(
            self.cfg_time["start"],
            self.cfg_time["format"],
        )
        self.end_t = datetime.strptime(
            self.cfg_time["end"],
            self.cfg_time["format"],
        )
        self.a_timestep = timedelta(
            hours=self.cfg_time['base_step_hours']
        )
        self.total_steps = ((self.end_t - self.start_t) // self.a_timestep) + 1

        # I/O control
        self.cfg_io = self.cfg["share"]["io_control"]
        self.base_dir = self.cfg_io["base_dir"]
        self.grib_dir = os.path.join(
            self.base_dir,
            self.cfg_io["grib_subdir"]
        )
        self.netcdf_dir = os.path.join(
            self.base_dir,
            self.cfg_io["netcdf_subdir"]
        )
        self.prefix = self.cfg_io["prefix"]
        self.pl_prefix = self.prefix["upper"]
        self.sl_prefix = self.prefix["surface"]
        #self.regrid_prefix = self.prefix["regrid"]
        self.output_prefix = self.prefix["output"]
        self.timestr_fmt = self.prefix["timestr_fmt"]
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)

        # preload target grids prevent from open file repeatly
        self.regrid = self.cfg["regrid"]
        self.target_nc = self.regrid["target_nc"]
        self.tgtlon = self.regrid["target_lon"]
        self.tgtlat = self.regrid["target_lat"]
        self.tgtpres = self.regrid["target_pres"]
        self.srclon = self.regrid["source_lon"]
        self.srclat = self.regrid["source_lat"]
        self.srcpres = self.regrid["source_pres"]
        self.pres_levels = self.regrid["levels"]
        self.adopted_varlist = self.regrid["adopted_varlist"]
        self.write_regrid = self.regrid["write_regrid"]
        with xr.open_dataset(self.target_nc, engine="netcdf4") as tgtds:
            self.XLONG = tgtds[self.tgtlon].values
            self.XLAT = tgtds[self.tgtlat].values
            self.static = tgtds[self.adopted_varlist]
            #self.outds = tgtds.copy(deep=True, data=data_vars["XLONG", "XLAT", "pres_levels"])

        # diagnostics module
        self.diagnostics = load_diagnostics(yaml_path)
        self.source_dataset = self.cfg["registry"]["source_dataset"]

    def _load_config(self, yaml_path):
        with open(yaml_path, mode="r") as f:
            return yaml.safe_load(f)

    def build_timeline(self):
        return [
            self.start_t + t * self.a_timestep
            for t in range(self.total_steps)
        ]

    def gen_io_filename(self, curr_time):
        timestamp = curr_time.strftime(self.timestr_fmt)
        pl_nc = f"{self.netcdf_dir}/{self.pl_prefix}_{timestamp}.nc"
        sl_nc = f"{self.netcdf_dir}/{self.sl_prefix}_{timestamp}.nc"
        #regrid_nc = f"{self.netcdf_dir}/{self.regrid_prefix}_{timestamp}.nc"
        output_nc = f"{self.netcdf_dir}/{self.output_prefix}_{timestamp}.nc"
        #print(pl_nc, "\n", sl_nc, "\n", regrid_nc, "\n", output_nc)
        return pl_nc, sl_nc, output_nc

    def _interpolate_with_fallback(self, points, values, xi):
        """
        Perform linear interpolation with a nearest-neighbor fallback for NaN values.

        Parameters
        ----------
        points : ndarray
            Coordinates of the source data points.
        values : ndarray
            Values of the source data points.
        xi : tuple
            Coordinates of the target grid.

        Returns
        -------
        ndarray
            Interpolated grid.
        """
        # First, try linear interpolation
        grid_linear = griddata(points, values, xi, method="linear")

        # Check if any NaNs were produced
        nan_mask = np.isnan(grid_linear)

        # If there are NaNs, use nearest neighbor to fill them
        if np.any(nan_mask):
            # Perform nearest interpolation
            grid_nearest = griddata(points, values, xi, method="nearest")
            # Fill in the NaNs from the linear result with values from the nearest result
            grid_linear[nan_mask] = grid_nearest[nan_mask]

        return grid_linear

    def interp_horizontal_v2(self, out_dict, curr_time, src_nc):
        """
        Interpolates data from a source grid to a target grid horizontally.
        Fills NaN values resulting from linear interpolation using the 'nearest' method.

        target_lat: "XLAT"
        target_lon: "XLONG"
        target_pres: "pres_levels"
        source_lat: "lat"
        source_lon: "lon"
        source_pres: "plev"

        Parameters
        ----------
        out_dict : dict
            Dictionary to store the output interpolated data.
        curr_time : datetime or similar
            Current time step being processed.
        src_nc : str or path-like
            Path to the source NetCDF file.

        Returns
        -------
        dict
            The updated dictionary with interpolated data.
        """
        dim_upp = ["Time", "pres_bottom_top", "south_north", "west_east"]
        dim_sfc = ["Time", "south_north", "west_east"]

        with xr.open_dataset(src_nc, engine="netcdf4") as ncds:
            lon = ncds[self.srclon].values
            lat = ncds[self.srclat].values
            if np.ndim(lon) == 1 and np.ndim(lat) == 1:
                lons, lats = np.meshgrid(lon, lat)
            else:
                lons, lats = lon, lat

            # Prepare source points for griddata
            points = np.vstack((lons.ravel(), lats.ravel())).T
            # Prepare target points
            xi = (self.XLONG.ravel(), self.XLAT.ravel())
            nt, ny, nx = self.XLONG.shape

            for var in ncds.keys():
                # Skip coordinate variables if they appear in the keys
                if var in [self.srclon, self.srclat, 'plev', 'time']:
                    continue

                data = np.squeeze(ncds[var].values)
                print(f"[REGRID] {var} => {data.shape}")

                if data.ndim == 3:
                    nl = data.shape[0]
                    data_h = np.empty((nl, ny, nx))
                    for pl in range(nl):
                        # Use the new interpolation function with fallback
                        interp_data = self._interpolate_with_fallback(
                            points, data[pl].ravel(), xi
                        )
                        data_h[pl] = np.reshape(interp_data, (ny, nx))

                    data_h = np.expand_dims(data_h, axis=0)
                    out_dict[var] = (dim_upp, data_h.astype(np.float32))

                elif data.ndim == 2:
                    # Use the new interpolation function with fallback
                    interp_data = self._interpolate_with_fallback(
                        points, data.ravel(), xi
                    )
                    data_h = np.reshape(interp_data, (ny, nx))
                    data_h = np.expand_dims(data_h, axis=0)

                    out_dict[var] = (dim_sfc, data_h.astype(np.float32))

        return out_dict

    def interp_horizontal(self, out_dict, curr_time, src_nc):
        """
        target_lat: "XLAT"
        target_lon: "XLONG"
        target_pres: "pres_levels"
        source_lat: "lat"
        source_lon: "lon"
        source_pres: "plev"
        Parameters
        ----------
        out_dict : TYPE
            DESCRIPTION.
        curr_time : TYPE
            DESCRIPTION.
        src_nc : TYPE
            DESCRIPTION.

        Returns
        -------
        interp_dict : TYPE
            DESCRIPTION.

        """
        dim_upp = ["Time", "pres_bottom_top", "south_north", "west_east"]
        dim_sfc = ["Time", "south_north", "west_east"]

        with xr.open_dataset(src_nc, engine="netcdf4") as ncds:
            lon = ncds[self.srclon].values
            lat = ncds[self.srclat].values
            if np.ndim(lon) == 1 and np.ndim(lat) == 1:
                lons, lats = np.meshgrid(lon, lat)
            else:
                lons, lats = lon, lat

            points = list(zip(lons.ravel(), lats.ravel()))
            xi = (self.XLONG, self.XLAT)
            nt, ny, nx = self.XLONG.shape

            for var in ncds.keys():
                data = np.squeeze(ncds[var].values)
                print(f"[REGRID] {var} => {data.shape}")
                if data.ndim == 3:
                    nl = data.shape[0]
                    data_h = np.empty((nl, ny, nx))
                    for pl in range(nl):
                        data_h[pl] = griddata(
                            points, data[pl].ravel(),
                            xi, method="linear"
                        )
                        # if np.isnan(data_h[pl]).any():
                        #     mean_mask = np.nanmean(data_h[pl].ravel())
                        #     data_h[pl][np.isnan(data_h[pl])] = mean_mask
                    data_h = np.expand_dims(data_h, axis=0)

                    out_dict[var] = (dim_upp, data_h.astype(np.float32))

                elif data.ndim == 2:
                    data_h = griddata(
                        points, data.ravel(), xi, method="linear"
                    )
                    # if np.isnan(data_h).any():
                    #     mean_mask = np.nanmean(data_h.ravel())
                    #     data_h[np.isnan(data_h)] = mean_mask
                    data_h = np.expand_dims(np.reshape(data_h, (ny,nx)), axis=0)

                    out_dict[var] = (dim_sfc, data_h.astype(np.float32))

        return out_dict


    def process_single_time(self, curr_time):
        print(f"[INFO] Processing single time: {curr_time}")
        [pl_nc, sl_nc, output_nc] = self.gen_io_filename(curr_time)

        out_dict = {}
        out2_dict = {}
        for var in self.static.data_vars: # Iterate over data_vars, not the Dataset itself
            static_data = self.static[var].values
            out_dict[var] = (self.static[var].dims, static_data) # Use original dimensions
            out2_dict[var] = (self.static[var].dims, static_data, self.static[var].attrs)

        # Ensure NetCDF files exist, otherwise skip this timestep
        if not os.path.exists(pl_nc) and not os.path.exists(sl_nc):
            print(f"[WARN] Missing NetCDF files for {curr_time}")
            return

        #out_dict = {}
        out_coords = {}
        nt, ny, nx = np.shape(self.XLONG)
        test_X = np.reshape(self.XLONG, (ny,nx))
        print(np.shape(test_X))
        #out_dict["XLONG"] = (
        #    ["Time", "south_north", "west_east"],
        #    np.expand_dims(np.reshape(self.XLONG,(ny,nx)))
        #)
        #out_dict["XLAT"] = (
        #    ["Time", "south_north", "west_east"], self.XLAT
        #)
        #out_dict["pres_levels"] = (
        #    ["pres_bottom_top"], self.pres_levels
        #)
        out_coords={
            "Time": ("Time", [np.datetime64(curr_time)]),
            "pres_bottom_top": ("pres_bottom_top", range(len(self.pres_levels))), # Dynamically get length
            "south_north": ("south_north", range(ny)),
            "west_east": ("west_east", range(nx)),
        }
        out2_coords = out_coords
        out_attrs={
            "title": f"Interpolated dataset at {curr_time}"
        }
        out2_attrs={
            "title": f"Diagnosed and interpolated dataset at {curr_time}"
        }


        # Horizontally interpolate pressure level data
        if os.path.exists(pl_nc):
            print("[REGRID]: ", pl_nc)
            self.interp_horizontal(out_dict, curr_time, pl_nc)

        # Horizontally interpolate surface level data
        if os.path.exists(sl_nc):
            print("[REGRID]: ", sl_nc)
            self.interp_horizontal(out_dict, curr_time, sl_nc)

        # Add static variables to the interpolation dictionary
        # Ensure correct dimensions for static variables;
        # assuming (Time, south_north, west_east) here
        #out2_dict = {}
        #for var in self.static.data_vars: # Iterate over data_vars, not the Dataset itself
        #    static_data = np.squeeze(self.static[var].values)
        #    out_dict[var] = (self.static[var].dims, static_data) # Use original dimensions
        #    out2_dict[var] = (self.static[var].dims, static_data, self.static[var].attrs)

        #nt, ny, nx = np.shape(self.XLONG)
        #out_dict["XLONG"] = (
        #    ["Time", "south_north", "west_east"],
        #    np.expand_dims(np.squeeze(self.XLONG), axis=0)
        #)
        #out_dict["XLAT"] = (
        #    ["Time", "south_north", "west_east"],
        #    np.expand_dims(np.squeeze(self.XLAT), axis=0)
        #)
        #out_dict["pres_levels"] = (
        #    ["pres_bottom_top"], self.pres_levels
        #)
        outds = xr.Dataset(
            data_vars = out_dict, coords = out_coords, attrs = out_attrs
        )
        out2ds = xr.Dataset(
            data_vars = out2_dict, coords = out2_coords, attrs = out2_attrs
        )
        if self.write_regrid:
            outds.to_netcdf(self.regrid_nc, format="NETCDF4")
            print(f"[DONE] Saved interpolated NetCDF for {curr_time}")
        else:
            print(f"[DONE] interpolated NetCDF for {curr_time} without saving the data")

        # --- Diagnostic variable calculation and output ---

        ordered_vars = sort_diagnostics_by_dependencies(self.diagnostics)
        #out2_dict = {}
        #out2_dict["XLONG"] = out_dict["XLONG"]
        #out2_dict["XLAT"] = out_dict["XLAT"]
        #out2_dict["pres_levels"] = out_dict["pres_levels"]


        for var in ordered_vars:
            if var not in self.diagnostics: # Skip variables that are just dependencies but not defined as outputs
                continue

            info = self.diagnostics[var]
            requires = info["requires"]
            #print("info requires = ", info, requires)
            diag_func = info["function"]

            if all(req in outds.data_vars or req in outds.coords for req in requires):
                print(f"[DIAGNOSE] Calculating diagnostic: {var}")
                try:
                    # === MODIFIED CALL ===
                    # Pass the source dataset type and the current dataset to the diagnostic function
                    diagnostic_dataarray = diag_func(self.source_dataset, outds)
                    # =====================

                    # Add the calculated diagnostic variable to the dataset
                    out2ds[var] = diagnostic_dataarray
                    print(f"[DIAGNOSE] Calculated {var}, shape: {out2ds[var].shape}, mean: {out2ds[var].values.mean():.4f}") # Access value after adding
                    #print(f"[DIAGNOSE] Calculated {var}, shape: {out2ds[var].shape}") # Simpler print

                except Exception as e:
                     print(f"[ERROR] Failed to calculate diagnostic {var}: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                # Find missing requirements
                missing = [req for req in requires if req not in outds.data_vars and req not in outds.coords]
                print(f"[WARN] Missing required inputs for diagnostic {var}: {missing}. Skipping calculation for this variable.")

        # Save once outside the loop
        out2ds.to_netcdf(output_nc, format="NETCDF4")
        print(f"[DONE] Saved diagnostic NetCDF for {curr_time}")

    def main_process(self):
        for curr_time in self.build_timeline():
            self.process_single_time(curr_time)
