# DLAMP.data

Streamlining Your Data-Driven Workflow: Pre-processing and Post-processing Utilities for DLAMP.tw Model.

This repository provides a suite of tools for pre-processing and post-processing data for the DLAMP.tw model. It includes functionalities for downloading data from the Climate Data Store (CDS), regridding data to a target domain, and calculating a wide range of diagnostic variables.

## How to install the python environment

### Condition 1. Simple Environment Setup for DLAMP.data

```bash
micromamba env create -n [envname] -c conda-forge python=3.11 conda python-cdo python-eccodes
pip install -r requirements.txt
```

### Condition 2. Environment Setup for DLAMP.tw and DLAMP.data [Experimental]

*   Please install DLAMP.tw first and freeze python version in 3.11
*   install hydra-core use extra "--upgrade" after installing the requirement records by pip
*   install onnxruntime according to your CUDA version, please check onnxruntime_official for more details.

```bash
micromamba env create -n [envname] -c conda-forge python=3.11 conda

git clone https://github.com/NVIDIA/physicsnemo && cd physicsnemo
make install && cd ..

git clone https://github.com/Chia-Tung/DLAMP DLAMP.tw && cd DLAMP.tw
pip install -r requirements.txt && \
pip install hydra-core --upgrade && \
pip install onnxruntime-gpu==1.20.0 && cd ..

git clone https://github.com/YaoChuDoSomething/DLAMP.data DLAMP.data && cd DLAMP.data
pip install -r requirement.txt
```

## How to Run the Workflow

The main entry point for the data processing workflow is `dlamp_prep.py`. You can control which parts of the workflow are executed by editing this file:

```python
###===== Workflow Control ===========================================###
#
###==================================================================###
from src.preproc.cds_downloader import CDSDataDownloader
from src.preproc.dlamp_regridder import DataRegridder

do_cds_downloader = 1  # Set to 1 to run the downloader, 0 to skip
do_dlamp_regridder = 1 # Set to 1 to run the regridder and diagnostics, 0 to skip

DLAMP_DATA_DIR = "./"
yaml_config = f"{DLAMP_DATA_DIR}/config/era5.yaml"
```

To run the workflow, simply execute the script:

```bash
python dlamp_prep.py
```

## Configuration

The entire workflow is controlled by YAML configuration files located in the `config/` directory. The main configuration file is `config/era5.yaml`.

### Controlling the Interpolation Mechanism

The interpolation (regridding) process is configured in the `regrid` section of `config/era5.yaml`.

```yaml
regrid:
  target_nc: "./assets/target.nc"         # Path to the NetCDF file defining the target grid
  target_lat: "XLAT"                      # Latitude variable name in the target file
  target_lon: "XLONG"                     # Longitude variable name in the target file
  target_pres: "pres_levels"              # Pressure level variable name in the target file
  source_lat: "lat"                       # Latitude variable name in the source data
  source_lon: "lon"                       # Longitude variable name in the source data
  source_pres: "plev"                     # Pressure level variable name in the source data
  levels: [1000, 975, ..., 20]            # List of pressure levels for vertical interpolation
  adopted_varlist: ["XLONG", "XLAT", "pres_levels", "HGT", "LANDMASK"] # Static variables to copy from target
  write_regrid: False                     # Set to True to save the intermediate regridded file
```

The system uses `scipy.interpolate.griddata` with a `"linear"` method for horizontal interpolation. You can modify the `src/preproc/dlamp_regridder.py` file, specifically the `interp_horizontal` method, to change the interpolation algorithm if needed.

### Controlling the Diagnostic Mechanism

The diagnostic variable calculation is controlled by the `registry` section in `config/era5.yaml`.

To **enable or disable** a diagnostic variable, simply **add or remove** its entry from this section.

```yaml
registry:
  source_dataset: "ERA5"
  varname:
    z_p:
      requires: ['z']
      function: diag_z_p

    tk_p:
      requires: ['t']
      function: diag_tk_p

    # ... other variables
```

-   `source_dataset`: Specifies the source of the data (e.g., "ERA5"). This is used by the diagnostic functions to apply the correct transformations.
-   `varname`: This is the dictionary containing all the diagnostic variables to be calculated.
-   Each entry under `varname` (e.g., `z_p`, `tk_p`) defines a diagnostic variable.
    -   `requires`: A list of source variables needed to calculate this diagnostic. The system will ensure these are available from the regridded data.
    -   `function`: The name of the Python function in `src/registry/diagnostic_functions.py` that performs the calculation.

### How to Add a New Diagnostic Variable and Method

Adding a new diagnostic variable is a three-step process:

**Step 1: Define the new variable in the configuration file.**

Open `config/era5.yaml` and add a new entry under `registry.varname`. For example, to add a new variable `my_new_var` that requires temperature (`t`) and surface pressure (`sp`):

```yaml
# in config/era5.yaml
registry:
  varname:
    # ... existing variables
    my_new_var:
      requires: ['t', 'sp']
      function: diag_my_new_var
```

**Step 2: Implement the calculation function.**

Open `src/registry/diagnostic_functions.py` and add a new Python function with the name you specified (`diag_my_new_var`). The function must accept `source_dataset` (a string) and `ds` (an `xarray.Dataset`) as arguments and return an `xarray.DataArray`.

```python
# in src/registry/diagnostic_functions.py

def diag_my_new_var(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Calculates my new variable.
    """
    # Example calculation
    temp = np.squeeze(ds["t"].values)
    sfc_pressure = np.squeeze(ds["sp"].values)

    # Perform your calculation
    data = temp * np.log(sfc_pressure)

    # Use the helper function to create a well-formed DataArray
    return _create_dataarray(
        data, ds, "my_new_var", "My New Diagnostic Variable", "units"
    )
```

**Step 3: The function is automatically registered.**

There is no need for a manual registration step. The system uses `importlib` to dynamically load the function specified in the YAML configuration from the `src.registry.diagnostic_functions` module.

### How the Mechanism Controls the Order of Diagnostics

The execution order of the diagnostic calculations is **not** determined by the order in `config/era5.yaml`. Instead, the system automatically determines the correct order based on the dependencies specified in the `requires` list for each variable.

The `src/registry/diagnostic_registry.py` file contains the `sort_diagnostics_by_dependencies` function. This function builds a dependency graph from the `requires` list of all variables and performs a topological sort. This ensures that if a diagnostic variable `B` requires the output of another diagnostic variable `A`, variable `A` will always be calculated before variable `B`.

This allows you to define variables in any order in the configuration file, and the system will intelligently execute them in the correct sequence.
