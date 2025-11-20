import os
import yaml
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
from loguru import logger

from earth2studio.data import GFS, CDS, fetch_data
from earth2studio.models.px import SFNO
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


class SFNODataProcessor:
    """
    Process SFNO global forecast model output and convert to ERA5-like format
    for downstream regional model preprocessing workflow.
    
    This class bridges SFNO forecasts with the existing DLAMP preprocessing pipeline,
    enabling seamless integration with DataRegridder for regional domain extraction.
    """
    
    def __init__(self, yaml_path):
        self.cfg = self._load_config(yaml_path)
        
        # Time control
        self.cfg_time = self.cfg["share"]["time_control"]
        self.start_t = datetime.strptime(
            self.cfg_time["start"],
            self.cfg_time["format"]
        )
        self.end_t = datetime.strptime(
            self.cfg_time["end"],
            self.cfg_time["format"]
        )
        self.forecast_hours = int((self.end_t - self.start_t).total_seconds() / 3600)
        
        # I/O control
        self.io = self.cfg["share"]["io_control"]
        self.base_dir = self.io["base_dir"]
        self.sfno_raw_dir = os.path.join(self.base_dir, self.io.get("sfno_raw_subdir", "./sfno_raw"))
        self.netcdf_dir = os.path.join(self.base_dir, self.io["netcdf_subdir"])
        self.prefix = self.io["prefix"]
        self.timestr_fmt = self.prefix['timestr_fmt']
        
        os.makedirs(self.sfno_raw_dir, exist_ok=True)
        os.makedirs(self.netcdf_dir, exist_ok=True)
        
        # SFNO configuration
        self.sfno_cfg = self.cfg.get("sfno", {})
        self.device = self._get_device()
        self.model = None
        self.data_source = None
        
        # Variable mapping: SFNO -> ERA5 naming convention
        self.var_mapping = self._build_variable_mapping()
        
        logger.info(f"SFNO Processor initialized for forecast: {self.start_t} to {self.end_t}")
    
    def _load_config(self, yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    
    def _get_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """Load SFNO model package"""
        if self.model is None:
            logger.info("Loading SFNO model package...")
            package = SFNO.load_default_package()
            self.model = SFNO.load_model(package).to(self.device)
            logger.info("SFNO model loaded successfully")
        return self.model
    
    def _get_data_source(self):
        """Initialize data source for initial conditions"""
        if self.data_source is None:
            source_name = self.sfno_cfg.get("initial_condition_source", "GFS")
            if source_name.upper() == "GFS":
                self.data_source = GFS()
            elif source_name.upper() == "CDS":
                self.data_source = CDS()
            else:
                raise ValueError(f"Unsupported data source: {source_name}")
            logger.info(f"Using {source_name} for initial conditions")
        return self.data_source
    
    def _build_variable_mapping(self):
        """
        Build mapping from SFNO output variables to ERA5 naming convention.
        This ensures compatibility with existing diagnostic functions.
        """
        # Default SFNO -> ERA5 variable mapping
        # Adjust based on actual SFNO output variable names
        mapping = {
            # Pressure level variables
            "z": "z",              # Geopotential
            "t": "t",              # Temperature
            "u": "u",              # U-component of wind
            "v": "v",              # V-component of wind
            "q": "q",              # Specific humidity
            "w": "w",              # Vertical velocity (omega)
            
            # Surface variables
            "t2m": "2t",           # 2m temperature
            "d2m": "2d",           # 2m dewpoint
            "u10": "10u",          # 10m u-wind
            "v10": "10v",          # 10m v-wind
            "sp": "sp",            # Surface pressure
            "msl": "msl",          # Mean sea level pressure
            "tp": "tp",            # Total precipitation
            "tcwv": "tcwv",        # Total column water vapor
        }
        
        # Allow override from config
        if "variable_mapping" in self.sfno_cfg:
            mapping.update(self.sfno_cfg["variable_mapping"])
        
        return mapping
    
    def run_forecast(self):
        """
        Run SFNO forecast and save raw output.
        Returns timeline of forecast valid times.
        """
        model = self._load_model()
        source = self._get_data_source()
        
        # Fetch initial conditions
        init_time = [self.start_t]
        logger.info(f"Fetching initial conditions for {self.start_t}")
        
        model_ic = model.input_coords()
        time_array = to_time_array(init_time)
        
        x, coords = fetch_data(
            source=source,
            time=time_array,
            variable=model_ic["variable"],
            lead_time=model_ic["lead_time"],
            device=self.device,
        )
        
        # Map coordinates
        x, coords = map_coords(x, coords, model.input_coords())
        
        # Run forecast
        nsteps = self.forecast_hours // 6  # SFNO outputs every 6 hours
        logger.info(f"Running SFNO forecast for {nsteps} steps ({self.forecast_hours} hours)")
        
        iterator = model.create_iterator(x, coords)
        forecast_data = []
        
        with tqdm(total=nsteps + 1, desc="SFNO Forecast") as pbar:
            for step, (x_step, coords_step) in enumerate(iterator):
                forecast_data.append({
                    'step': step,
                    'data': x_step.cpu().numpy(),
                    'coords': coords_step
                })
                pbar.update(1)
                if step == nsteps:
                    break
        
        logger.info("SFNO forecast completed")
        
        # Save raw forecast to netCDF
        self._save_raw_forecast(forecast_data, init_time[0])
        
        # Return timeline
        return self.create_timeline()
    
    def _save_raw_forecast(self, forecast_data, init_time):
        """Save raw SFNO forecast output"""
        output_file = os.path.join(
            self.sfno_raw_dir,
            f"sfno_forecast_{init_time.strftime('%Y%m%d_%H%M')}.nc"
        )
        
        # Convert to xarray Dataset (simplified - adjust based on actual structure)
        logger.info(f"Saving raw SFNO forecast to {output_file}")
        # Implementation depends on SFNO output structure
        # This is a placeholder for the actual conversion logic
    
    def create_timeline(self):
        """Create timeline of forecast valid times"""
        base_step_hours = self.cfg_time.get("base_step_hours", 1)
        a_timestep = timedelta(hours=base_step_hours)
        total_steps = int(self.forecast_hours / base_step_hours) + 1
        
        return [
            self.start_t + t * a_timestep
            for t in range(total_steps)
        ]
    
    def convert_to_era5_format(self, curr_time):
        """
        Convert SFNO forecast at given time to ERA5-like format.
        This creates separate upper-level and surface files compatible with DataRegridder.
        
        Parameters
        ----------
        curr_time : datetime
            The valid time to extract from SFNO forecast
            
        Returns
        -------
        tuple
            Paths to (pressure_level_file, surface_level_file)
        """
        timestamp = curr_time.strftime(self.timestr_fmt)
        
        pl_nc = os.path.join(
            self.netcdf_dir,
            f"{self.prefix.get('sfno_upper', 'sfnopl')}_{timestamp}.nc"
        )
        sl_nc = os.path.join(
            self.netcdf_dir,
            f"{self.prefix.get('sfno_surface', 'sfnosl')}_{timestamp}.nc"
        )
        
        # Skip if already converted
        if os.path.exists(pl_nc) and os.path.exists(sl_nc):
            logger.info(f"SFNO data already converted for {curr_time}")
            return pl_nc, sl_nc
        
        # Load raw SFNO forecast
        forecast_file = os.path.join(
            self.sfno_raw_dir,
            f"sfno_forecast_{self.start_t.strftime('%Y%m%d_%H%M')}.nc"
        )
        
        if not os.path.exists(forecast_file):
            logger.error(f"Raw SFNO forecast not found: {forecast_file}")
            return None, None
        
        logger.info(f"Converting SFNO forecast to ERA5 format for {curr_time}")
        
        with xr.open_dataset(forecast_file) as ds:
            # Extract forecast at specific time
            lead_hours = int((curr_time - self.start_t).total_seconds() / 3600)
            
            # Split into pressure level and surface variables
            pl_vars, sl_vars = self._split_variables(ds, lead_hours)
            
            # Rename variables to ERA5 convention
            pl_vars = self._rename_variables(pl_vars)
            sl_vars = self._rename_variables(sl_vars)
            
            # Save to separate files
            pl_vars.to_netcdf(pl_nc, format="NETCDF4")
            sl_vars.to_netcdf(sl_nc, format="NETCDF4")
            
            logger.info(f"Converted SFNO data saved: {pl_nc}, {sl_nc}")
        
        return pl_nc, sl_nc
    
    def _split_variables(self, ds, lead_hours):
        """
        Split SFNO output into pressure level and surface variables.
        This mirrors the ERA5 data structure.
        """
        # Select data at specific lead time
        if 'lead_time' in ds.dims:
            ds_time = ds.sel(lead_time=lead_hours, method='nearest')
        else:
            ds_time = ds
        
        # Identify pressure level variables (3D) vs surface (2D)
        pl_var_list = []
        sl_var_list = []
        
        for var in ds_time.data_vars:
            if 'plev' in ds_time[var].dims or 'level' in ds_time[var].dims:
                pl_var_list.append(var)
            else:
                sl_var_list.append(var)
        
        pl_vars = ds_time[pl_var_list] if pl_var_list else xr.Dataset()
        sl_vars = ds_time[sl_var_list] if sl_var_list else xr.Dataset()
        
        return pl_vars, sl_vars
    
    def _rename_variables(self, ds):
        """Rename variables from SFNO to ERA5 convention"""
        rename_dict = {k: v for k, v in self.var_mapping.items() if k in ds.data_vars}
        return ds.rename(rename_dict)
    
    def process_forecast_for_regridding(self):
        """
        Main workflow: Run SFNO forecast and convert all timesteps to ERA5 format.
        This prepares data for the DataRegridder pipeline.
        """
        # Step 1: Run SFNO forecast
        timeline = self.run_forecast()
        
        # Step 2: Convert each timestep to ERA5 format
        logger.info(f"Converting {len(timeline)} timesteps to ERA5 format")
        
        converted_files = []
        for curr_time in tqdm(timeline, desc="Converting to ERA5 format"):
            pl_nc, sl_nc = self.convert_to_era5_format(curr_time)
            if pl_nc and sl_nc:
                converted_files.append((curr_time, pl_nc, sl_nc))
        
        logger.info(f"Conversion complete. {len(converted_files)} timesteps ready for regridding.")
        return converted_files


class SFNOFeedbackInterface:
    """
    Interface for two-way data exchange between global SFNO and regional model.
    
    This class manages:
    1. Downscaling: SFNO global -> Regional domain (via DataRegridder)
    2. Upscaling: Regional model output -> SFNO boundary update (future capability)
    """
    
    def __init__(self, yaml_path):
        self.cfg = self._load_config(yaml_path)
        self.feedback_cfg = self.cfg.get("feedback", {})
        self.enabled = self.feedback_cfg.get("enabled", False)
        
        if self.enabled:
            logger.info("Two-way feedback interface enabled")
            self._initialize_feedback()
        else:
            logger.info("Two-way feedback interface disabled")
    
    def _load_config(self, yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    
    def _initialize_feedback(self):
        """Initialize feedback mechanisms"""
        self.update_frequency = self.feedback_cfg.get("update_frequency_hours", 6)
        self.feedback_vars = self.feedback_cfg.get("feedback_variables", [])
        self.blending_method = self.feedback_cfg.get("blending_method", "linear")
        self.blending_zone_width = self.feedback_cfg.get("blending_zone_width_km", 100)
        
        logger.info(f"Feedback configuration: update every {self.update_frequency}h, "
                   f"blending method: {self.blending_method}")
    
    def downscale_to_regional(self, sfno_data, regional_domain):
        """
        Downscale SFNO global forecast to regional domain.
        This is the primary operational mode.
        
        Parameters
        ----------
        sfno_data : xr.Dataset
            SFNO forecast data in ERA5-like format
        regional_domain : dict
            Domain specification (north, south, west, east bounds)
            
        Returns
        -------
        xr.Dataset
            Regional domain data ready for high-resolution modeling
        """
        logger.info("Downscaling SFNO forecast to regional domain")
        
        # Extract regional domain from global forecast
        regional_data = sfno_data.sel(
            lat=slice(regional_domain['south'], regional_domain['north']),
            lon=slice(regional_domain['west'], regional_domain['east'])
        )
        
        return regional_data
    
    def upscale_regional_to_global(self, regional_data, global_grid):
        """
        Upscale regional model output to global grid for SFNO boundary update.
        This is for future two-way nesting capability.
        
        Parameters
        ----------
        regional_data : xr.Dataset
            High-resolution regional model output
        global_grid : xr.Dataset
            Target global grid specification
            
        Returns
        -------
        xr.Dataset
            Regional data interpolated to global grid with blending
        """
        if not self.enabled:
            logger.warning("Feedback interface is disabled. Upscaling skipped.")
            return None
        
        logger.info("Upscaling regional model output to global grid (placeholder)")
        
        # Placeholder for future implementation:
        # 1. Interpolate regional data to global grid resolution
        # 2. Apply blending at boundaries
        # 3. Merge with SFNO forecast outside regional domain
        
        return None  # To be implemented
    
    def apply_boundary_blending(self, regional_data, global_data, domain_spec):
        """
        Apply smooth blending at regional domain boundaries.
        Prevents sharp discontinuities in nested configurations.
        """
        if not self.enabled:
            return regional_data
        
        logger.info(f"Applying boundary blending with {self.blending_method} method")
        
        # Placeholder for blending implementation
        # This would create a transition zone where:
        # blended = α * regional + (1-α) * global
        # where α varies from 1 (interior) to 0 (boundary)
        
        return regional_data
    
    def update_global_forecast(self, sfno_forecast, regional_feedback, update_time):
        """
        Update SFNO global forecast with regional model feedback.
        Future capability for improved global forecasts.
        
        Parameters
        ----------
        sfno_forecast : xr.Dataset
            Current SFNO global forecast
        regional_feedback : xr.Dataset
            Regional model output to assimilate
        update_time : datetime
            Time of the update
            
        Returns
        -------
        xr.Dataset
            Updated global forecast
        """
        if not self.enabled:
            logger.info("Two-way feedback disabled, returning original SFNO forecast")
            return sfno_forecast
        
        logger.info(f"Updating global forecast at {update_time} (placeholder)")
        
        # Placeholder for future implementation:
        # 1. Upscale regional data
        # 2. Apply data assimilation or blending
        # 3. Reinitialize SFNO with updated fields
        
        return sfno_forecast