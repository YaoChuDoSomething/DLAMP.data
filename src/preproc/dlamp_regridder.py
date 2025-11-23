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
        self.base_timestep = timedelta(
            hours=self.cfg_time['base_step_hours']
        )
        self.output_timestep = timedelta(
            hours=self.cfg["regrid"]['output_step_hours']
        )
        self.total_steps = ((self.end_t - self.start_t) // self.output_timestep) + 1

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

    def build_input_timeline(self):
        total_input_steps = ((self.end_t - self.start_t) // self.base_timestep) + 1
        return [
            self.start_t + t * self.base_timestep 
            for t in range(total_input_steps)
        ]

    def build_output_timeline(self):
        return [
            self.start_t + t * self.output_timestep 
            for t in range(self.total_steps)
        ]

    def gen_io_filename(self, curr_time):
        timestamp = curr_time.strftime(self.timestr_fmt)
        input_nc = f"{self.netcdf_dir}/{self.output_prefix}_{timestamp}.nc"
        output_nc = f"{self.netcdf_dir}/{self.output_prefix}_diag_{timestamp}.nc"
        #print(pl_nc, "\n", sl_nc, "\n", regrid_nc, "\n", output_nc)
        return input_nc, output_nc

    def interp_horizontal(self, out_dict, curr_time, src_ds):
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
        src_ds : xarray.Dataset
            The source dataset object for interpolation.

        Returns
        -------
        interp_dict : TYPE
            DESCRIPTION.

        """
        dim_upp = ["Time", "pres_bottom_top", "south_north", "west_east"]
        dim_sfc = ["Time", "south_north", "west_east"]
        
        lon = src_ds[self.srclon].values
        lat = src_ds[self.srclat].values
        if np.ndim(lon) == 1 and np.ndim(lat) == 1:
            lons, lats = np.meshgrid(lon, lat)
        else:
            lons, lats = lon, lat
        
        points = list(zip(lons.ravel(), lats.ravel()))
        xi = (self.XLONG, self.XLAT)
        nt, ny, nx = self.XLONG.shape

        for var in src_ds.keys():
            data = np.squeeze(src_ds[var].values)
            print(f"[REGRID] {var} => {data.shape}")
            if data.ndim == 3:
                nl = data.shape[0]
                data_h = np.empty((nl, ny, nx))
                for pl in range(nl):
                    data_h[pl] = griddata(
                        points, data[pl].ravel(), 
                        xi, method="linear"
                    )
                data_h = np.expand_dims(data_h, axis=0)
            
                out_dict[var] = (dim_upp, data_h.astype(np.float32))
            
            elif data.ndim == 2:
                data_h = griddata(
                    points, data.ravel(), xi, method="linear"
                )
                data_h = np.expand_dims(np.reshape(data_h, (ny,nx)), axis=0)
                
                out_dict[var] = (dim_sfc, data_h.astype(np.float32))
                
        return out_dict
    
    
    def process_single_time(self, curr_time):
        print(f"[INFO] Processing single time: {curr_time}")
        input_nc, output_nc = self.gen_io_filename(curr_time)
        
        # Ensure NetCDF files exist, otherwise skip this timestep
        if not os.path.exists(input_nc):
            print(f"[WARN] Missing NetCDF file for {curr_time}: {input_nc}")
            return

        out_dict = {}

        # Horizontally interpolate data from the single input file
        print("[REGRID]: ", input_nc)
        self.interp_horizontal(out_dict, curr_time, input_nc)

        # Add static variables to the interpolation dictionary
        # Ensure correct dimensions for static variables; 
        # assuming (Time, south_north, west_east) here
        for var in self.static.data_vars: # Iterate over data_vars, not the Dataset itself
            static_data = self.static[var].values
            print(var)
            # Ensure static data is compatible with time dimension
            if static_data.ndim == 2: # Assuming static variables are (south_north, west_east)
                out_dict[var] = (
                    ("Time", "south_north", "west_east"), 
                    np.expand_dims(static_data, axis=0),
                )
            else:
                out_dict[var] = (self.static[var].dims, static_data) # Use original dimensions

        nt, ny, nx = np.shape(self.XLONG)
        out_dict["XLONG"] = (
            ["Time", "south_north", "west_east"], 
            np.expand_dims(np.squeeze(self.XLONG), axis=0)
        )
        out_dict["XLAT"] = (
            ["Time", "south_north", "west_east"], 
            np.expand_dims(np.squeeze(self.XLAT), axis=0)
        )
        out_dict["pres_levels"] = (
            ["pres_bottom_top"], self.pres_levels
        )
        outds = xr.Dataset(
            data_vars=out_dict,
            coords={
                "Time": ("Time", [pd.Timestamp(curr_time)]),
                "pres_bottom_top": ("pres_bottom_top", self.pres_levels), # Dynamically get length
                "south_north": ("south_north", np.arange(ny)),
                "west_east": ("west_east", np.arange(nx)),
            },
            attrs={
                "title": f"Interpolated dataset at {curr_time}"
            }
        )
        if self.write_regrid:
            outds.to_netcdf(self.regrid_nc, format="NETCDF4")
            print(f"[DONE] Saved interpolated NetCDF for {curr_time}")
        else:
            print(f"[DONE] interpolated NetCDF for {curr_time} without saving the data")
        
        # --- Diagnostic variable calculation and output ---
        
        ordered_vars = sort_diagnostics_by_dependencies(self.diagnostics)

        for var in ordered_vars:
            if var not in self.diagnostics: # Skip variables that are just dependencies but not defined as outputs
                continue

            info = self.diagnostics[var]
            requires = info["requires"]
            diag_func = info["function"]
            
            if all(req in outds.data_vars or req in outds.coords for req in requires):
                #print(f"[DIAGNOSE] Calculating diagnostic: {var}")
                try:
                    # === MODIFIED CALL ===
                    # Pass the source dataset type and the current dataset to the diagnostic function
                    diagnostic_dataarray = diag_func(self.source_dataset, outds)
                    # =====================

                    # Add the calculated diagnostic variable to the dataset
                    outds[var] = diagnostic_dataarray
                    # print(f"[DIAGNOSE] Calculated {var}, shape: {outds[var].shape}, mean: {outds[var].values.mean():.4f}") # Access value after adding
                    print(f"[DIAGNOSE] Calculated {var}, shape: {outds[var].shape}") # Simpler print

                except Exception as e:
                     print(f"[ERROR] Failed to calculate diagnostic {var}: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                # Find missing requirements
                missing = [req for req in requires if req not in outds.data_vars and req not in outds.coords]
                print(f"[WARN] Missing required inputs for diagnostic {var}: {missing}. Skipping calculation for this variable.")
        
        # Save once outside the loop
        outds.to_netcdf(output_nc, format="NETCDF4")
        print(f"[DONE] Saved diagnostic NetCDF for {curr_time}")

    def temporal_interp(self, time_t, ds1, ds2):
        """
        Linearly interpolate variables between two datasets at a specific time.
        """
        # Calculate the time difference (fraction) for interpolation
        t1 = ds1.Time.values[0]
        t2 = ds2.Time.values[0]
        time_frac = (time_t - t1) / (t2 - t1)

        # Create a new dataset for the interpolated data
        interp_ds = xr.Dataset()

        # Interpolate each variable
        for var_name in ds1.data_vars:
            if var_name in ds2.data_vars:
                # Linear interpolation: var1 + (var2 - var1) * time_frac
                interp_data = ds1[var_name] + (ds2[var_name] - ds1[var_name]) * time_frac
                interp_ds[var_name] = interp_data

        # Update the time coordinate of the new dataset
        interp_ds = interp_ds.assign_coords(Time=[time_t])
        return interp_ds

    def main_process(self):
        input_timeline = self.build_input_timeline()
        output_timeline = self.build_output_timeline()

        for t_idx, curr_time in enumerate(output_timeline):
            print(f"[INFO] Processing output time: {curr_time}")

            # Find surrounding input files for interpolation
            prev_input_time = max([t for t in input_timeline if t <= curr_time])
            next_input_time_list = [t for t in input_timeline if t > curr_time]
            
            if not next_input_time_list:
                print(f"[WARN] No next input time found for {curr_time}. End of data range.")
                # If it's the very last timestamp, just process it directly
                if curr_time == prev_input_time:
                    input_nc, _ = self.gen_io_filename(curr_time)
                    if os.path.exists(input_nc):
                         with xr.open_dataset(input_nc) as ds:
                            self.process_and_save(ds, curr_time)
                continue

            next_input_time = next_input_time_list[0]
            
            # Get filenames for the two closest input times
            prev_input_nc, _ = self.gen_io_filename(prev_input_time)
            next_input_nc, _ = self.gen_io_filename(next_input_time)

            if not os.path.exists(prev_input_nc) or not os.path.exists(next_input_nc):
                print(f"[WARN] Missing one or both input files for interpolation at {curr_time}. Skipping.")
                continue

            # Open datasets
            with xr.open_dataset(prev_input_nc) as ds1, xr.open_dataset(next_input_nc) as ds2:
                # If the current time is an exact match with an input file, no interpolation needed
                if curr_time == prev_input_time:
                    interp_ds = ds1.copy(deep=True)
                else:
                    # Perform temporal interpolation
                    interp_ds = self.temporal_interp(curr_time, ds1, ds2)
                
                # Process and save the (potentially interpolated) dataset
                self.process_and_save(interp_ds, curr_time)
    
    def process_and_save(self, dataset, curr_time):
        print(f"[INFO] Processing and saving for time: {curr_time}")
        _, output_nc = self.gen_io_filename(curr_time)

        out_dict = {}

        # Horizontally interpolate data from the dataset
        self.interp_horizontal(out_dict, curr_time, dataset)
        
        # --- The rest of the processing logic from the old process_single_time ---
        
        # Add static variables to the interpolation dictionary
        for var in self.static.data_vars:
            static_data = self.static[var].values
            if static_data.ndim == 2:
                out_dict[var] = (("Time", "south_north", "west_east"), np.expand_dims(static_data, axis=0))
            else:
                out_dict[var] = (self.static[var].dims, static_data)

        nt, ny, nx = np.shape(self.XLONG)
        out_dict["XLONG"] = (["Time", "south_north", "west_east"], np.expand_dims(np.squeeze(self.XLONG), axis=0))
        out_dict["XLAT"] = (["Time", "south_north", "west_east"], np.expand_dims(np.squeeze(self.XLAT), axis=0))
        out_dict["pres_levels"] = (["pres_bottom_top"], self.pres_levels)

        outds = xr.Dataset(
            data_vars=out_dict,
            coords={
                "Time": ("Time", [pd.Timestamp(curr_time)]),
                "pres_bottom_top": ("pres_bottom_top", self.pres_levels),
                "south_north": ("south_north", np.arange(ny)),
                "west_east": ("west_east", np.arange(nx))
            },
            attrs={"title": f"Interpolated dataset at {curr_time}"}
        )

        if self.write_regrid:
            regrid_nc, _ = self.gen_io_filename(curr_time)
            # We need a different name for the regridded file if we save it
            regrid_output_nc = regrid_nc.replace(".nc", "_regridded.nc")
            outds.to_netcdf(regrid_output_nc, format="NETCDF4")
            print(f"[DONE] Saved interpolated NetCDF for {curr_time}")

        # --- Diagnostic variable calculation ---
        ordered_vars = sort_diagnostics_by_dependencies(self.diagnostics)
        for var in ordered_vars:
            if var not in self.diagnostics:
                continue
            info = self.diagnostics[var]
            requires = info["requires"]
            diag_func = info["function"]
            if all(req in outds.data_vars or req in outds.coords for req in requires):
                try:
                    diagnostic_dataarray = diag_func(self.source_dataset, outds)
                    outds[var] = diagnostic_dataarray
                    print(f"[DIAGNOSE] Calculated {var}, shape: {outds[var].shape}")
                except Exception as e:
                    print(f"[ERROR] Failed to calculate diagnostic {var}: {e}")
            else:
                missing = [req for req in requires if not (req in outds.data_vars or req in outds.coords)]
                print(f"[WARN] Missing inputs for {var}: {missing}. Skipping.")
        
        outds.to_netcdf(output_nc, format="NETCDF4")
        print(f"[DONE] Saved final diagnostic NetCDF for {curr_time}")
        
    
    
# if __name__ == "__main__":
#     yaml_file = "/wk2/yaochu/RESEARCH/dlamp/DLAMP.data.beta/config/DLAMPreproc.yaml"
#     regridder = DataRegridder(yaml_file)
#     regridder.main_process()
    

    #yaml_file = "/wk2/yaochu/RESEARCH/dlamp/DLAMP.data.beta/config/DLAMPreproc.yaml"
    #cfg = yaml.safe_load(yaml_file)

    #regridder = DataRegridder(yaml_file)
    
    #interpolater.main_process()
    #att = interpolater.build_nc_attributes()
    #df = pd.read_csv("../assets/rwrf_Vtable.csv")
    

        
    
    
    
