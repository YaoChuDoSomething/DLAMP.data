# src/workflows/sfno_forecast.py
import yaml
import os
from datetime import datetime, timedelta
import xarray as xr
from earth2studio.models.px import SFNO

# Add project root to the Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.workflows import data_processing

def main(config, config_path):
    """
    Main function for the SFNO forecast workflow.
    """
    print("Executing SFNO forecast workflow.")
    
    # 1. Load configuration
    time_control = config["share"]["time_control"]
    ic_config_path = config["initial_condition"]["config"]
    
    # 2. Fetch initial condition using the data_processing workflow
    print(f"Fetching initial condition using config: {ic_config_path}")
    with open(ic_config_path, 'r') as f:
        ic_config = yaml.safe_load(f)
    
    # Load the target grid once
    regrid_config = ic_config["regrid"]
    with xr.open_dataset(regrid_config["target_nc"], autoclose=True) as target_ds:
        target_lon = target_ds[regrid_config["target_lon"]].squeeze().values
        target_lat = target_ds[regrid_config["target_lat"]].squeeze().values

    # Run the data processing workflow, passing in the pre-loaded target grid
    data_processing.main(ic_config, ic_config_path, target_lon=target_lon, target_lat=target_lat)
    
    # Construct the path to the initial condition file
    start_time = datetime.strptime(time_control["start"], time_control["format"])
    ic_io_config = ic_config["share"]["io_control"]
    timestamp = start_time.strftime(ic_io_config["prefix"]["timestr_fmt"])
    ic_filename = f"{ic_io_config['prefix']['output']}_{timestamp}.nc"
    ic_path = os.path.join(ic_io_config["base_dir"], ic_io_config["netcdf_subdir"], ic_filename)
    
    print(f"Loading initial condition from: {ic_path}")
    initial_condition_ds = xr.open_dataset(ic_path)
    
    # 3. Transform initial condition data into SFNO's required DataArray format (Placeholder)
    print("Transforming initial condition for SFNO... (Not yet implemented)")
    # This will involve selecting the 73 variables and stacking them.
    
    # 4. Load the SFNO model
    print("Loading SFNO model...")
    try:
        model = SFNO.from_pretrained() # Using a default SFNO model
        print("SFNO model loaded successfully.")
    except Exception as e:
        print(f"Failed to load SFNO model: {e}")
        return

    # 5. Forecast loop (Placeholder)
    print("Starting forecast loop... (Not yet implemented)")
    
    forecast_steps = time_control["forecast_steps"]
    current_time = start_time
    for i in range(forecast_steps):
        print(f"Forecasting step {i+1}/{forecast_steps} for time {current_time}")
        
        # a. (Task 3) Blending (Not yet implemented)
        # b. Run SFNO model (Not yet implemented)
        # c. Post-process output (Not yet implemented)
        # d. Run diagnostics (Not yet implemented)
        # e. Save output (Not yet implemented)
        
        current_time += timedelta(hours=time_control["forecast_step_hours"])

    print("SFNO forecast workflow finished.")