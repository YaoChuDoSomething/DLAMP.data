# Earth2Studio-Based Workflow Integration Plan

## 1.0 Overall Objective

To refactor the existing data processing pipeline (`DLAMPreproc.py`) and introduce new forecasting capabilities by leveraging the `earth2studio` library. This plan outlines three core tasks:
1.  **Task 1:** A unified data ingestion workflow to process initial conditions from both CDS (ERA5) and GFS, and convert them into RWRF format.
2.  **Task 2:** A one-way forecasting workflow where the SFNO model is initialized with GFS data to produce a forecast, which is then converted to RWRF format.
3.  **Task 3 (Tentative):** A two-way coupled workflow where the SFNO global forecast is updated at each time step with data from a high-resolution regional model.

This refactoring aims to create a modular, configurable, and extensible system for weather data processing and forecasting, while adhering to strict project structure and data handling conventions.

## 2.0 Project Structure & Conventions

To maintain consistency and modularity, the project will adhere to the following structure and conventions:

1.  **Entry Point:** A single entry script, `main.py`, will be located in the project root. This script will act as a controller, parsing arguments to select and execute the desired workflow (e.g., data processing, SFNO forecast).
2.  **Configuration:** All configuration files will be located in the `config/` directory. New YAML files, such as `config/gfs.yaml` and `config/sfno.yaml`, will be created to manage parameters for their respective workflows.
3.  **Code Organization:** All new Python source code, excluding the main entry point, will be placed within the `src/` directory. A new subdirectory, `src/workflows/`, will be created to house the logic for the different tasks.
4.  **Data Structure:** Workflows will primarily use `xarray.Dataset` for handling meteorological data. For interaction with `earth2studio` models like SFNO that expect a "channel" or "variable" dimension, the data will be transformed into an `xarray.DataArray` with coordinates like `("variable", "lat", "lon")` or `("variable", "south_north", "west_east")` as required.
5.  **Code Modification Constraints:**
    -   **Forbidden:** No modifications will be made to files within `src/preproc/` or to `src/registry/diagnostic_registry.py`.
    -   **Allowed:** New diagnostic functions can be added to `src/registry/diagnostic_functions.py` to support new variables or data sources.

## 3.0 Task 1: Data Ingestion and RWRF Conversion (CDS & GFS)

### 3.1 Objective
Create a robust and unified workflow that can fetch data from different sources (initially CDS/ERA5 and GFS), perform necessary interpolations, and convert it to the RWRF format using the existing diagnostic tools.

### 3.2 Workflow Steps
1.  **Configuration Loading:** The workflow will be initiated by loading a source-specific configuration file (e.g., `config/era5.yaml` or `config/gfs.yaml`).
2.  **Data Acquisition:**
    -   For ERA5 data, `earth2studio.data.CDS` will be used.
    -   For GFS data, `earth2studio.data.GFS` will be used.
    -   The data will be fetched directly as `xarray` objects, eliminating the need for intermediate GRIB/NetCDF files.
3.  **Regridding:** The fetched data, regardless of its source grid, will be horizontally interpolated to the target RWRF grid defined in `assets/target.nc`. The output coordinates will be `("south_north", "west_east")`.
4.  **Diagnostics and Formatting:** The regridded `xarray.Dataset` will be passed to the existing diagnostic system. The functions in `src/registry/diagnostic_functions.py` (called in the correct order determined by `sort_diagnostics_by_dependencies`) will calculate the required RWRF variables.
5.  **Output:** The final `xarray.Dataset`, now in RWRF format, will be saved as a NetCDF file, following the naming conventions specified in the configuration.

## 4.0 Task 2: One-Way SFNO-GFS Forecast Workflow

### 4.1 Objective
Implement a complete forecast-to-RWRF pipeline. This workflow will use GFS data as the initial condition for an SFNO deep learning weather model and process the multi-step forecast output into RWRF-formatted NetCDF files.

### 4.2 Workflow Steps
1.  **Fetch Initial Condition:** Use the workflow defined in **Task 1** to acquire and process GFS data for the forecast's initial time (`t0`).
2.  **Data Transformation for SFNO:**
    -   The initial condition `xarray.Dataset` will be transformed into the `xarray.DataArray` format required by the SFNO model.
    -   This involves selecting the 73 required input variables (u10m, v10m, t2m, etc.), ensuring they are in the correct order, and stacking them along a new `variable` coordinate.
3.  **Load SFNO Model:** The pre-trained SFNO model will be loaded using the `earth2studio.models.load()` interface.
4.  **Generate Forecast:** The workflow will iterate through the desired number of forecast time steps. In each step, the SFNO model will be called to predict the state at the next time step.
5.  **Post-process and Save Output:** For each forecast time step generated by the model:
    -   The output `xarray.DataArray` will be converted back into an `xarray.Dataset` with standard variable names.
    -   The data will be regridded to the RWRF grid (if the model's native grid is different).
    -   The RWRF diagnostic functions will be run to produce the final set of variables.
    -   The resulting RWRF-formatted `xarray.Dataset` will be saved to a NetCDF file.

## 5.0 Task 3: Two-Way Coupled Workflow (Regional -> Global)

### 5.1 Objective (Tentative)
Enhance the SFNO forecast by incorporating higher-resolution data from a regional model. Before each forecast step, the global state will be updated with the regional forecast, allowing the global model to benefit from fine-grained regional details.

### 5.2 Modified Workflow
This task modifies the forecast loop described in **Task 2**. The process for a single forecast step from `t` to `t+1` is as follows:

1.  **Load Regional Data:** Load the forecast output from the regional model valid at time `t`.
2.  **Data Assimilation/Update:**
    -   Identify the geographical domain where the regional and global grids overlap.
    -   Perform a smooth blending operation to update the SFNO global state (the input for the current step) with the regional data. This may involve techniques like feathering at the boundaries of the regional domain to prevent sharp discontinuities.
3.  **Run SFNO Forecast:** Execute the SFNO model for one time step (`t -> t+1`) using the blended, updated state as input.
4.  **Post-process Output:** The output from the SFNO model is then post-processed into RWRF format as described in Task 2.
5.  **Loop:** This updated state for `t+1` becomes the base for the next iteration's blending process.

### 5.3 Key Challenges
The primary challenge is the implementation of a numerically stable and effective blending algorithm. Care must be taken to smoothly merge the two data sources without introducing artifacts that could degrade the forecast quality.

## 6.0 Implementation Plan

1.  **Refactor Entry Point:** Create the new `main.py` script in the root directory to handle workflow selection.
2.  **Add Configurations:** Create `config/gfs.yaml` and `config/sfno.yaml` with the necessary parameters for data fetching, model selection, and I/O control.
3.  **Develop Workflows:**
    -   Implement the unified data processing logic (Task 1) in `src/workflows/data_processing.py`.
    -   Implement the SFNO forecasting and coupling logic (Task 2 & 3) in `src/workflows/sfno_forecast.py`.
4.  **Validation:** Thoroughly validate each workflow.
    -   **Task 1:** Compare the RWRF output from the new workflow against the output from the original `DLAMPreproc.py` to ensure consistency.
    -   **Task 2 & 3:** Analyze the forecast outputs for physical plausibility and stability.
