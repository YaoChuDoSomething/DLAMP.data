# src/workflows/data_processing.py
import yaml
import os
from datetime import datetime, timedelta
from earth2studio.data import GFS
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# Add project root to the Python path to allow importing from src
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.registry.diagnostic_registry import load_diagnostics, sort_diagnostics_by_dependencies

def _interpolate_with_fallback(points, values, xi):
    """
    Perform linear interpolation with a nearest-neighbor fallback for NaN values.
    """
    grid_linear = griddata(points, values, xi, method="linear")
    nan_mask = np.isnan(grid_linear)
    if np.any(nan_mask):
        grid_nearest = griddata(points, values, xi, method="nearest")
        grid_linear[nan_mask] = grid_nearest[nan_mask]
    return grid_linear

def regrid_data(source_da: xr.DataArray, target_lon: np.ndarray, target_lat: np.ndarray) -> xr.Dataset:
    """
    Regrids a source DataArray to a target grid defined by target_lon and target_lat.
    """
    # Prepare source grid
    source_lon = source_da['lon'].values
    source_lat = source_da['lat'].values
    if source_lon.ndim == 1 and source_lat.ndim == 1:
        source_lon, source_lat = np.meshgrid(source_lon, source_lat)
    
    points = np.vstack((source_lon.ravel(), source_lat.ravel())).T
    
    # Prepare target grid
    xi = (target_lon.ravel(), target_lat.ravel())
    
    regridded_vars = {}
    # Use a dictionary to build the new dataset to avoid repeated concatenation
    for var_name in source_da['variable'].values:
        print(f"Regridding variable: {var_name}")
        source_var_data = source_da.sel(variable=var_name).values.ravel()
        
        # Interpolate
        interp_data = _interpolate_with_fallback(points, source_var_data, xi)
        regridded_var = np.reshape(interp_data, target_lon.shape)
        
        regridded_vars[var_name] = (('south_north', 'west_east'), regridded_var)

    # Create the Dataset
    regridded_ds = xr.Dataset(
        regridded_vars,
        coords={
            'XLONG': (('south_north', 'west_east'), target_lon),
            'XLAT': (('south_north', 'west_east'), target_lat)
        }
    )
    return regridded_ds

def run_diagnostics(ds: xr.Dataset, config_path: str, config: dict) -> xr.Dataset:
    """
    Runs diagnostic calculations on the dataset.
    """
    diagnostics = load_diagnostics(config_path)
    ordered_vars = sort_diagnostics_by_dependencies(diagnostics)
    source_dataset_type = config.get("registry", {}).get("source_dataset", "GFS")

    output_ds = ds.copy()

    for var in ordered_vars:
        if var not in diagnostics:
            continue

        info = diagnostics[var]
        requires = info["requires"]
        diag_func = info["function"]

        if all(req in output_ds.data_vars or req in output_ds.coords for req in requires):
            print(f"[DIAGNOSE] Calculating diagnostic: {var}")
            try:
                diagnostic_dataarray = diag_func(source_dataset_type, output_ds)
                output_ds[var] = diagnostic_dataarray
            except Exception as e:
                print(f"[ERROR] Failed to calculate diagnostic {var}: {e}")
        else:
            missing = [req for req in requires if req not in output_ds.data_vars and req not in output_ds.coords]
            print(f"[WARN] Missing required inputs for diagnostic {var}: {missing}. Skipping.")
            
    return output_ds

def main(config, config_path, target_lon=None, target_lat=None):
    """
    Main function for the data processing workflow.
    """
    print("Executing data processing workflow.")
    
    # 1. Load configuration
    data_source_name = config.get("data_source", {}).get("name", "").lower()
    time_control = config["share"]["time_control"]
    regrid_config = config["regrid"]
    io_config = config["share"]["io_control"]
    
    # Load target grid if not provided
    if target_lon is None or target_lat is None:
        print("Loading target grid from file.")
        with xr.open_dataset(regrid_config["target_nc"], autoclose=True) as target_ds:
            target_lon = target_ds[regrid_config["target_lon"]].squeeze().values
            target_lat = target_ds[regrid_config["target_lat"]].squeeze().values
    
    start_time = datetime.strptime(time_control["start"], time_control["format"])
    end_time = datetime.strptime(time_control["end"], time_control["format"])
    time_step_hours = time_control["base_step_hours"]
    
    current_time = start_time
    
    # 2. Initialize data source
    if data_source_name == "gfs":
        source_options = config.get("data_source", {})
        data_source = GFS(
            source=source_options.get("source", "aws"),
            cache=source_options.get("cache", True)
        )
        print(f"Initialized data source: {data_source_name}")
    else:
        raise ValueError(f"Unsupported data source: {data_source_name}")

    # 3. Fetch and process data for each timestep
    while current_time <= end_time:
        print(f"Processing time: {current_time}")
        
        try:
            # Fetch data
            variables = config["data_source"]["variables"]
            data_da = data_source(time=[current_time], variable=variables)
            print(f"Successfully fetched data for {current_time}")

            # Regrid data
            regridded_ds = regrid_data(data_da.squeeze('time'), target_lon, target_lat)
            print(f"Successfully regridded data for {current_time}")

            # Run diagnostics
            diagnosed_ds = run_diagnostics(regridded_ds, config_path, config)
            print(f"Successfully ran diagnostics for {current_time}:")
            print(diagnosed_ds)

            # Save output
            output_dir = os.path.join(io_config["base_dir"], io_config["netcdf_subdir"])
            os.makedirs(output_dir, exist_ok=True)
            timestamp = current_time.strftime(io_config["prefix"]["timestr_fmt"])
            output_filename = f"{io_config['prefix']['output']}_{timestamp}.nc"
            output_path = os.path.join(output_dir, output_filename)
            
            diagnosed_ds.to_netcdf(output_path, format="NETCDF4")
            print(f"Successfully saved output to {output_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to process data for {current_time}: {e}")
        
        current_time += timedelta(hours=time_step_hours)

    print("Data processing workflow finished.")