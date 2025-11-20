# SFNO Global Forecast Integration for DLAMP Regional Model

## Overview

This module integrates the **SFNO (Spherical Fourier Neural Operator)** global weather forecast model into the DLAMP regional model preprocessing pipeline. It enables seamless conversion of SFNO global forecasts to RWRF-compatible format for regional high-resolution modeling.

### Key Features

- **Global-to-Regional Workflow**: SFNO global forecast → ERA5-like format → Regional domain extraction → RWRF format
- **Variable Compatibility**: Automatic conversion of SFNO variables to ERA5/RWRF naming conventions
- **Flexible Configuration**: YAML-based configuration for all workflow stages
- **Two-Way Feedback Interface**: Future capability for regional-global model coupling
- **Diagnostic Variables**: Comprehensive set of derived meteorological variables

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SFNO Preprocessing Workflow               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: SFNO Forecast                                      │
│  - Load SFNO model (earth2studio)                           │
│  - Fetch initial conditions (GFS/CDS)                       │
│  - Run global forecast (6-hourly output)                    │
│  - Save raw SFNO output → ./sfno_raw/                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Format Conversion                                  │
│  - Convert SFNO variables to ERA5 naming                    │
│  - Split into pressure level / surface files                │
│  - Output: sfnopl_*.nc, sfnosl_*.nc → ./ncdb/Pool/         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Regridding to Regional Domain                     │
│  - Horizontal interpolation (global → regional grid)        │
│  - Use existing DataRegridder class                         │
│  - Output: sfnoregrid_*.nc                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Diagnostic Variables                              │
│  - Calculate derived variables (Q2, rh2, etc.)              │
│  - Emulate radar reflectivity                               │
│  - Output: sfnodlamp_*.nc (RWRF-compatible)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Optional: Two-Way Feedback (Future)                        │
│  - Regional model output → Global boundary update           │
│  - Boundary blending and data assimilation                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

```bash
# Required packages
pip install earth2studio torch xarray netcdf4 scipy pyyaml loguru tqdm
```

### Directory Structure

```
DLAMP_model/
├── DLAMP.data/
│   ├── sfno_raw/              # Raw SFNO forecast output
│   ├── ncdb/Pool/             # Converted ERA5-like format
│   ├── grib/                  # Optional GRIB files
│   └── npy/                   # Optional numpy arrays
├── config/
│   ├── sfno.yaml              # SFNO configuration
│   └── era5.yaml              # ERA5 configuration (reference)
├── src/
│   ├── preproc/
│   │   ├── sfno_processor.py     # SFNO forecast & conversion
│   │   ├── cds_downloader.py     # ERA5 downloader
│   │   └── dlamp_regridder.py    # Regridding engine
│   └── registry/
│       ├── diagnostic_functions.py   # Diagnostic calculations
│       └── diagnostic_registry.py    # Diagnostic registry
├── assets/
│   └── target.nc              # Regional domain specification
├── SFNOPreproc.py             # Main workflow script
├── DLAMPreproc.py             # ERA5 workflow script
└── README_SFNO.md             # This file
```

---

## Quick Start

### 1. Configure YAML Settings

Edit `config/sfno.yaml`:

```yaml
share:
  time_control:
    start: "2024-10-01_00:00"    # Forecast initialization
    end: "2024-10-03_00:00"      # Forecast end (48-hour forecast)
    format: "%Y-%m-%d_%H:%M"
    base_step_hours: 1           # Output timestep
  
  io_control:
    base_dir: "/path/to/DLAMP.data/"
    # ... (see full config file)

sfno:
  initial_condition_source: "GFS"  # or "CDS"
  # ... (see full config file)
```

### 2. Run Full Workflow

```bash
# Run complete preprocessing pipeline
python SFNOPreproc.py --config config/sfno.yaml

# With custom log directory
python SFNOPreproc.py --config config/sfno.yaml --log-dir ./logs/sfno
```

### 3. Run Specific Stages

```bash
# Run only SFNO forecast (skip conversion/regridding)
python SFNOPreproc.py --stages forecast

# Skip forecast, run conversion and regridding (use existing SFNO output)
python SFNOPreproc.py --skip-forecast --stages convert,regrid,diagnostics

# Run only regridding and diagnostics
python SFNOPreproc.py --skip-forecast --stages regrid,diagnostics
```

---

## Configuration Details

### Time Control

```yaml
time_control:
  start: "2024-10-01_00:00"      # Forecast initialization time
  end: "2024-10-03_00:00"        # Forecast valid time (48h forecast)
  format: "%Y-%m-%d_%H:%M"       # Datetime format
  base_step_hours: 1             # Output interval (1h interpolated from 6h)
```

### SFNO Model Settings

```yaml
sfno:
  initial_condition_source: "GFS"    # Options: "GFS", "CDS"
  model_step_hours: 6                # SFNO native timestep
  
  # Variables to extract from SFNO forecast
  output_variables:
    pressure_level:
      - "z"    # Geopotential
      - "t"    # Temperature
      - "u"    # U-wind
      - "v"    # V-wind
      - "q"    # Specific humidity
      - "w"    # Omega (vertical velocity)
    
    surface:
      - "t2m"   # 2m temperature
      - "d2m"   # 2m dewpoint
      - "u10"   # 10m u-wind
      - "v10"   # 10m v-wind
      - "sp"    # Surface pressure
      - "msl"   # Mean sea level pressure
      - "tp"    # Total precipitation
      - "tcwv"  # Total column water vapor
```

### Regional Domain

```yaml
regrid:
  target_nc: "./assets/target.nc"     # Regional grid specification
  target_lat: "XLAT"                  # Target latitude variable
  target_lon: "XLONG"                 # Target longitude variable
  target_pres: "pres_levels"          # Pressure levels
  
  levels: [1000, 975, 950, ..., 20]   # Vertical levels (hPa)
```

---

## Variable Mapping

SFNO outputs are automatically converted to ERA5-like naming conventions:

| SFNO Variable | ERA5 Name | Description |
|---------------|-----------|-------------|
| `t` | `t` | Temperature (K) |
| `z` | `z` | Geopotential (m²/s²) |
| `u` | `u` | U-wind (m/s) |
| `v` | `v` | V-wind (m/s) |
| `q` | `q` | Specific humidity (kg/kg) |
| `w` | `w` | Omega (Pa/s) |
| `t2m` | `2t` | 2m temperature (K) |
| `d2m` | `2d` | 2m dewpoint (K) |
| `u10` | `10u` | 10m u-wind (m/s) |
| `v10` | `10v` | 10m v-wind (m/s) |
| `sp` | `sp` | Surface pressure (Pa) |
| `msl` | `msl` | Mean sea level pressure (Pa) |
| `tp` | `tp` | Total precipitation (m) |
| `tcwv` | `tcwv` | Total column water vapor (kg/m²) |

---

## Diagnostic Variables

The following derived variables are calculated:

### Pressure Level Variables
- `z_p`: Geopotential height (m) = z / 9.80665
- `tk_p`: Air temperature (K)
- `umet_p`, `vmet_p`: Wind components (m/s)
- `QVAPOR_p`: Water vapor mixing ratio (kg/kg) = q / (1-q)
- `wa_p`: Vertical velocity (m/s) from omega

### Hydrometeor Variables (set to zero for SFNO)
- `QRAIN_p`, `QCLOUD_p`, `QSNOW_p`, `QICE_p`, `QGRAUP_p`
- `QTOTAL_p`: Total hydrometeors

### Surface Variables
- `T2`: 2m temperature (K)
- `Q2`: 2m mixing ratio (kg/kg)
- `rh2`: 2m relative humidity (%)
- `td2`: 2m dewpoint temperature (K)
- `umet10`, `vmet10`: 10m winds (m/s)
- `slp`: Sea level pressure (hPa)
- `PSFC`: Surface pressure (Pa)
- `pw`: Precipitable water (kg/m²)
- `RAINNC`: Total precipitation (m)

### Optional Variables (if available in SFNO)
- `SST`: Sea surface temperature (K)
- `PBLH`: Planetary boundary layer height (m)
- `SWDOWN`: Surface shortwave radiation (W/m²)
- `OLR`: Outgoing longwave radiation (W/m²)
- `REFL`: Radar reflectivity (dBZ) - minimal for SFNO

---

## Two-Way Feedback Interface (Future Feature)

### Current Status: **Placeholder Implementation**

The feedback interface is designed for future two-way nesting between SFNO and regional models:

```yaml
feedback:
  enabled: False                         # Currently disabled
  update_frequency_hours: 6              # How often to update global
  blending_method: "linear"              # Boundary blending method
  blending_zone_width_km: 100            # Transition zone width
  feedback_variables:                    # Variables to feed back
    - "t"
    - "q"
    - "u"
    - "v"
```

### Planned Capabilities

1. **Downscaling** (Currently Implemented)
   - Extract regional domain from SFNO global forecast
   - Interpolate to high-resolution regional grid
   - Provide boundary conditions for regional model

2. **Upscaling** (Future Implementation)
   - Interpolate regional model output to global grid
   - Apply boundary blending to avoid discontinuities
   - Update SFNO forecast with regional improvements

3. **Boundary Blending** (Future Implementation)
   - Smooth transition at domain boundaries
   - Methods: linear, cosine taper, relaxation
   - Configurable blending zone width

4. **Data Assimilation** (Future Implementation)
   - Assimilate regional model output into SFNO
   - Improve global forecast accuracy
   - Enable true two-way coupling

---

## Comparison with ERA5 Workflow

| Feature | ERA5 (DLAMPreproc.py) | SFNO (SFNOPreproc.py) |
|---------|----------------------|----------------------|
| Data Source | CDS API (reanalysis) | SFNO model (forecast) |
| Temporal Coverage | Historical + recent | Forecast only |
| Spatial Resolution | ~31 km (0.25°) | ~100 km (1.0°) |
| Vertical Levels | 31 levels | 31 levels (configurable) |
| Temporal Resolution | 1-hourly | 6-hourly (interpolated to 1h) |
| Hydrometeors | Yes (ERA5 has crwc, clwc, etc.) | No (set to zero) |
| Lead Time | N/A (analysis) | Up to 10-14 days |
| Computational Cost | Download only | GPU inference required |
| Use Case | Initial conditions, validation | Boundary conditions, forecasting |

---

## Performance Considerations

### Memory Requirements
- **SFNO Model**: ~4-8 GB GPU memory
- **Data Processing**: ~2-4 GB RAM per timestep
- **Storage**: ~500 MB per timestep (all variables)

### Runtime Estimates (48-hour forecast, RTX 3090)
- SFNO Forecast: ~5-10 minutes
- Format Conversion: ~2-5 minutes
- Regridding (450x450 domain): ~10-20 minutes
- Diagnostics: ~5-10 minutes
- **Total**: ~25-45 minutes

### Optimization Tips
1. **Use GPU**: Essential for SFNO inference
2. **Batch Processing**: Process multiple timesteps together
3. **Parallel Regridding**: Use multiprocessing for independent timesteps
4. **Disk I/O**: Use fast SSD for intermediate files
5. **Memory Management**: Clear large arrays after processing

---

## Troubleshooting

### Common Issues

#### 1. SFNO Model Loading Fails
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU (slower)
```python
device = torch.device("cpu")  # Force CPU
```

#### 2. Variable Name Mismatch
```
Error: Variable 't2m' not found in dataset
```
**Solution**: Check SFNO output variable names and update mapping in config

#### 3. Regridding NaN Values
```
Warning: NaN values in interpolated grid
```
**Solution**: Uses automatic fallback to nearest-neighbor interpolation

#### 4. Missing Hydrometeor Fields
```
Warning: SFNO doesn't provide hydrometeor fields
```
**Expected**: SFNO sets QRAIN, QCLOUD, etc. to zero (not an error)

---

## Advanced Usage

### Custom Initial Conditions

```python
from src.preproc.sfno_processor import SFNODataProcessor

processor = SFNODataProcessor("config/sfno.yaml")

# Use custom initial condition dataset
processor.data_source = MyCustomDataSource()
timeline = processor.run_forecast()
```

### Modify Diagnostic Functions

Add new diagnostics in `src/registry/diagnostic_functions.py`:

```python
def diag_MY_VARIABLE(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Custom diagnostic variable"""
    match source_dataset:
        case "SFNO":
            # Calculate from SFNO variables
            data = calculate_my_variable(ds)
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "MY_VAR", "Description", "units")
```

Then add to `config/sfno.yaml`:

```yaml
registry:
  varname:
    MY_VAR:
      requires: ['t', 'q']
      function: diag_MY_VARIABLE
```

---

## Future Enhancements

1. **Multi-Model Ensemble**: Support for multiple global models (SFNO, GraphCast, etc.)
2. **Real-Time Forecasting**: Automatic daily forecast generation
3. **Two-Way Coupling**: Full regional-global feedback implementation
4. **Bias Correction**: ML-based post-processing of SFNO forecasts
5. **Verification Tools**: Forecast skill metrics and validation
6. **Data Assimilation**: Integrate observations into SFNO forecasts
7. **Cloud Deployment**: Containerized workflow for scalability

---

## References

- **Earth2Studio Documentation**: https://nvidia.github.io/earth2studio/
- **SFNO Paper**: Bonev et al. (2023), "Spherical Fourier Neural Operators"
- **DLAMP Project**: Regional Deep Learning Atmospheric Model
- **ERA5 Documentation**: ECMWF Reanalysis v5

---

## Support

For questions or issues:
1. Check documentation and troubleshooting section
2. Review log files in `./logs/`
3. Open issue on project repository
4. Contact DLAMP development team

---

## License

This module is part of the DLAMP project and follows the same license terms.

---

## Changelog

### Version 1.0.0 (2024-11)
- Initial implementation
- SFNO forecast integration
- ERA5-compatible format conversion
- Regional domain regridding
- Diagnostic variable calculation
- Two-way feedback interface (placeholder)