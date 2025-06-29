import yaml
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from registry.diagnostic_registry import load_diagnostics, sort_diagnostics_by_dependencies

from datetime import datetime, timedelta
import os

class DataRegridder:
    """
    A class to regrid atmospheric data, process it, and calculate diagnostics.

    This class handles loading configuration from a YAML file, managing time control
    for data processing, performing horizontal interpolation of atmospheric variables,
    and calculating diagnostic variables based on defined dependencies.

    Attributes:
        cfg (dict): The loaded configuration from the YAML file.
        cfg_time (dict): Time control settings from the configuration.
        start_t (datetime): The start time for processing.
        end_t (datetime): The end time for processing.
        a_timestep (timedelta): The time step for processing.
        total_steps (int): The total number of time steps to process.
        cfg_io (dict): File I/O settings from the configuration.
        base_dir (str): The base directory for data.
        grib_dir (str): Directory for GRIB files.
        netcdf_dir (str): Directory for NetCDF files.
        prefix (dict): Filename prefixes.
        pl_prefix (str): Prefix for pressure level NetCDF files.
        sl_prefix (str): Prefix for surface level NetCDF files.
        regrid_prefix (str): Prefix for regridded NetCDF files.
        output_prefix (str): Prefix for output diagnostic NetCDF files.
        timestr_fmt (str): Timestamp format for filenames.
        regrid (dict): Regridding settings.
        target_nc (str): Path to the target NetCDF file for regridding.
        xaxis (str): Variable name for longitude in the target grid.
        yaxis (str): Variable name for latitude in the target grid.
        paxis (str): Variable name for pressure levels in the target grid.
        levels (list): Pressure levels to consider.
        adopted_varlist (list): List of adopted variables for regridding.
        XLONG (np.ndarray): Longitude values of the target grid.
        XLAT (np.ndarray): Latitude values of the target grid.
        static (xarray.Dataset): Static variables from the target grid.
        diagnostics (dict): Loaded diagnostic functions and their metadata.
    """
    def __init__(self, yaml_path):
        """
        Initializes the DataRegridder with a YAML configuration file.

        Args:
            yaml_path (str): The path to the YAML configuration file.
        """
        # Load YAML configuration file
        self.cfg = self._load_config(yaml_path)

        # Time control
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
        self.cfg_io = self.cfg["share"]["file_io"]
        self.base_dir = self.cfg_io["base_dir"]
        self.grib_dir = os.path.join(self.base_dir, self.cfg_io["grib_subdir"])
        self.netcdf_dir = os.path.join(self.base_dir, self.cfg_io["netcdf_subdir"])
        self.prefix = self.cfg_io["fn_prefix"]
        self.pl_prefix = self.prefix["plev"]
        self.sl_prefix = self.prefix["slev"]
        self.regrid_prefix = self.prefix["regrid"]
        self.output_prefix = self.prefix["output"]
        self.timestr_fmt = self.prefix["timestr_fmt"]
        os.makedirs(self.grib_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)

        # Preload target grids to prevent repeated file opening
        self.regrid = self.cfg["regrid"]
        self.target_nc = self.regrid["target_nc"]
        self.tgtlon = self.regrid["target_lon"]
        self.tgtlat = self.regrid["target_lat"]
        self.tgtpres = self.regrid["target_pres"]
        self.srclon = self.regrid["source_lon"]
        self.srclat = self.regrid["source_lat"]
        self.srcpres = self.regrid["source_pres"]
        self.levels = self.regrid["levels"]
        self.adopted_varlist = self.regrid["adopted_varlist"]
        self.write_nc = self.regrid["write_nc"]
        with xr.open_dataset(self.target_nc, engine="netcdf4") as tgtds:
            self.XLONG = tgtds[self.tgtlon].values
            self.XLAT = tgtds[self.tgtlat].values
            self.static = tgtds[self.adopted_varlist]

        # Diagnostics module
        self.diagnostics = load_diagnostics(yaml_path)


    def _load_config(self, yaml_path):
        """
        Loads the YAML configuration file.

        Args:
            yaml_path (str): The path to the YAML configuration file.

        Returns:
            dict: The loaded configuration dictionary.
        """
        with open(yaml_path, mode="r") as f:
            return yaml.safe_load(f)

    def build_timeline(self):
        """
        Builds a list of datetime objects representing the processing timeline.

        Returns:
            list: A list of datetime objects.
        """
        return [
            self.start_t + t * self.a_timestep
            for t in range(self.total_steps)
        ]

    def gen_io_filename(self, curr_time):
        """
        Generates input/output filenames for a given time.

        Args:
            curr_time (datetime): The current datetime for which to generate filenames.

        Returns:
            tuple: A tuple containing the paths to pressure level NetCDF,
                   surface level NetCDF, regridded NetCDF, and output NetCDF files.
        """
        timestamp = curr_time.strftime(self.timestr_fmt)
        pl_nc = f"{self.netcdf_dir}/{self.pl_prefix}_{timestamp}.nc"
        sl_nc = f"{self.netcdf_dir}/{self.sl_prefix}_{timestamp}.nc"
        regrid_nc = f"{self.netcdf_dir}/{self.regrid_prefix}_{timestamp}.nc"
        output_nc = f"{self.netcdf_dir}/{self.output_prefix}_{timestamp}.nc"
        return pl_nc, sl_nc, regrid_nc, output_nc

    def interp_horizontal(self, interp_dict, curr_time, src_nc, xlon, xlat):
        """
        Performs horizontal interpolation of data from a source NetCDF file
        to a target grid.

        Args:
            interp_dict (dict): A dictionary to store interpolated data.
            curr_time (datetime): The current time for logging.
            src_nc (str): Path to the source NetCDF file.
            xlon (np.ndarray): Longitude values of the target grid.
            xlat (np.ndarray): Latitude values of the target grid.

        Returns:
            dict: The updated dictionary with interpolated data.
        """
        with xr.open_dataset(src_nc, engine="netcdf4") as ncds:
            lon = ncds[self.srclon].values
            lat = ncds[self.srclat].values
            if np.ndim(lon) == 1 and np.ndim(lat) == 1:
                lons, lats = np.meshgrid(lon, lat)
            else:
                lons, lats = lon, lat

            points = list(zip(lons.ravel(), lats.ravel()))
            xi = (xlon, xlat)
            ny, nx = xlon.shape

            dim3d = ("Time", "pres_bottom_top", "south_north", "west_east")
            dim2d = ("Time", "south_north", "west_east")

            for var in ncds.data_vars:
                data = np.squeeze(ncds[var].values)
                print(ncds[var].dims)
                print(curr_time, " => ", var, " = ", np.shape(data))
                if data.ndim == 3:
                    nl = data.shape[0]
                    data_h = np.empty((nl, ny, nx))
                    for pl in range(nl):
                        d = data[pl].ravel()
                        data_h[pl] = griddata(points, d, xi, method="linear")
                    interp_dict[var] = (dim3d, np.expand_dims(data_h, axis=0).astype(float))
                elif data.ndim == 2:
                    d = data.ravel()
                    data_h = griddata(points, d, xi, method="linear")
                    interp_dict[var] = (dim2d, np.expand_dims(data_h, axis=0).astype(float))

        return interp_dict

    def process_single_time(self, curr_time):
        """
        Processes data for a single time step, including interpolation and diagnostic calculation.

        Args:
            curr_time (datetime): The current datetime to process.
        """
        print(f"[INFO] Processing single time: {curr_time}")
        pl_nc, sl_nc, regrid_nc, output_nc = self.gen_io_filename(curr_time)

        # Ensure NetCDF files exist, otherwise skip this timestep
        if not os.path.exists(pl_nc) and not os.path.exists(sl_nc):
            print(f"[WARN] Missing input NetCDF files for {curr_time}, skipping.")
            return

        interp_dict = {}

        # Horizontally interpolate pressure level data
        if os.path.exists(pl_nc)
            interp_dict = self.interp_horizontal(
                interp_dict,
                curr_time,
                pl_nc,
                self.XLONG,
                self.XLAT,
            )

        # Horizontally interpolate surface level data
        if os.path.exists(sl_nc)
            interp_dict = self.interp_horizontal(
                interp_dict,
                curr_time,
                sl_nc,
                self.XLONG,
                self.XLAT,
            )

        # Add static variables to the interpolation dictionary
        # Ensure correct dimensions for static variables;
        # assuming (Time, south_north, west_east) here
        for var in self.static.data_vars: # Iterate over data_vars, not the Dataset itself
            static_data = self.static[var].values
            print(var)
            # Ensure static data is compatible with time dimension
            if static_data.ndim == 2: # Assuming static variables are (south_north, west_east)
                interp_dict[var] = (
                    ("Time", "south_north", "west_east"),
                    np.expand_dims(static_data, axis=0),
                )
            else:
                interp_dict[var] = (self.static[var].dims, static_data) # Use original dimensions

        # Add pressure level information
        with xr.open_dataset(pl_nc, engine="netcdf4") as plds:
            plev = plds["plev"].values

        interp_dict["plev"] = (("pres_bottom_top"), plev)

        ny, nx = np.shape(self.XLONG)
        interp_ds = xr.Dataset(
            data_vars=interp_dict,
            coords={
                "Time": ("Time", [pd.Timestamp(curr_time)]),
                "pres_bottom_top": ("pres_bottom_top", plev), # Dynamically get length
                "south_north": ("south_north", np.arange(ny)),
                "west_east": ("west_east", np.arange(nx)),
            },
            attrs={
                "title": f"Interpolated dataset at {curr_time}"
            }
        )

        if self.write_nc:
            interp_ds.to_netcdf(regrid_nc, format="NETCDF4")
            print(f"[DONE] Saved interpolated NetCDF for {curr_time}")
        else:
            print(f"[DONE] interpolated NetCDF for {curr_time} without saving the data")

        # --- Diagnostic variable calculation and output ---

        ordered_vars = sort_diagnostics_by_dependencies(self.diagnostics)

        # Initialize a new Dataset to store diagnostic results, including time and spatial coordinates
        # Coordinates can be copied from interp_ds or redefined
        output_ds = xr.Dataset(
            coords={
                "Time": ("Time", [pd.Timestamp(curr_time)]),
                "pres_bottom_top": ("pres_bottom_top", interp_ds["pres_bottom_top"].values),
                "south_north": ("south_north", interp_ds["south_north"].values),
                "west_east": ("west_east", interp_ds["west_east"].values),
            },
            attrs={
                "title": f"Diagnostic output at {curr_time}"
            }
        )

        # Directly add static variables to output_ds
        for var in self.static.data_vars:
            output_ds[var] = interp_ds[var] # Directly copy from interp_ds

        for var in ordered_vars:
            info = self.diagnostics[var]
            requires = info["requires"]

            # Check if all required variables are in interp_ds
            if all(req in interp_ds.data_vars for req in requires):
                # Pass the entire interp_ds to the diagnostic function
                # Directly call the diagnostic function, which now returns an xr.DataArray
                diagnostic_dataarray = info["function"](interp_ds)

            # Add the returned DataArray directly to output_ds
            # xarray will automatically handle dimension and coordinate matching
                output_ds[var] = diagnostic_dataarray
                print(f"[INFO] Calculated diagnostic: {var}")
            else:
                print(f"[WARN] Missing required inputs for diagnostic {var}: {requires}, skip.")

        # Save once outside the loop
        output_ds.to_netcdf(output_nc, format="NETCDF4")
        print(f"[DONE] Saved diagnostic NetCDF for {curr_time}")

    def main_process(self):
        """
        The main processing loop that iterates through the timeline
        and processes data for each time step.
        """
        for curr_time in self.build_timeline():
            self.process_single_time(curr_time)

#HOW-TO
#if __name__ == "__main__":
#    yaml_file = "./config/DLAMPreproc.yaml"
#    regridder = DataRegridder(yaml_file)
#    regridder.main_process()
