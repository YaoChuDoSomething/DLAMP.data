# `src/op` Module Structure

This directory (`src/op`) contains the core operational components of the DLAMP data processing pipeline, refactored to adhere strictly to the Single Responsibility Principle (SRP) and designed for a configuration-driven workflow. Each Python file encapsulates a distinct set of functionalities, making the system modular, testable, and maintainable.

## Module Breakdown

### `era5_reanalysis.py`
*   **Purpose**: Manages the retrieval and initial processing of ERA5 reanalysis data from the CDS API. It handles downloading GRIB files, merging them, converting to NetCDF, and basic data validation.
*   **Key Classes/Functions**:
    *   `DataPipelineError`, `CDSRetrievalError`, `ProcessingError`, `ValidationError`: Custom exceptions for pipeline stages.
    *   `CDSRetriever`: Handles direct interaction with the CDS API for data download.
    *   `GribProcessor`: Manages GRIB file operations, including merging multiple GRIBs into a single NetCDF and cleaning up temporary files. Relies on `cdo`.
    *   `DataValidator`: Performs basic checks on the generated NetCDF files to ensure data integrity and presence of required variables/coordinates.
    *   `ERA5DataManager`: Orchestrates the entire ERA5 data preparation process, ensuring both surface and pressure level data are correctly downloaded, merged, and validated into a single NetCDF file for model input.

### `global_operators.py`
*   **Purpose**: Contains functionalities related to global model operations, specifically designed for models like SFNO, supporting both one-way downscaling and two-way coupling workflows. It focuses on data loading, variable standardization, model inference, feedback coupling, and data export.
*   **Key Classes/Functions**:
    *   `GlobalOperatorsError`: Custom exception for errors in global operations.
    *   `VariableMapper`: Standardizes variable names and dimensions for model compatibility, decoupled from file I/O.
    *   `ModelLoader`: Handles loading model weights (e.g., SFNO) and raw NetCDF file I/O.
    *   `InferenceEngine`: Executes model inference in a generator-based fashion, allowing external control for step-by-step processing, crucial for coupled systems.
    *   `FeedbackCoupler`: Implements logic for blending regional model feedback into global fields, supporting two-way coupling.
    *   `DataExporter`: Manages data output, including extracting specific regions for downscaling and saving NetCDF files.

### `regridders.py`
*   **Purpose**: Provides a comprehensive pipeline for data regridding and transformation, essential for preparing global model outputs for regional models or vice-versa. It covers variable standardization, temporal upsampling, vertical coordinate transformation, and optimized spatial regridding.
*   **Key Classes/Functions**:
    *   `PreprocessingError`: Base exception for regridding errors.
    *   `DataStandardizer`: Standardizes variable names, units, and dimensions.
    *   `TimeUpsampler`: Handles temporal interpolation (e.g., 6-hourly to 1-hourly data).
    *   `VerticalTransformer`: Transforms vertical coordinates, such as calculating log-pressure height.
    *   `SpatialRegridder`: Optimizes spatial interpolation from global to regional grids by pre-slicing data with a buffer, then interpolating to a target grid (supporting 2D lat/lon coordinates).
    *   `DataPreparationPipeline`: Orchestrates the sequence of standardization, temporal, vertical, and spatial transformations.

### `diagnostics.py`
*   **Purpose**: Calculates a wide range of diagnostic meteorological variables from raw model outputs or reanalysis data. It separates physical constants, thermodynamic calculations, data building, and the diagnostic engine itself.
*   **Key Classes/Functions**:
    *   `MetConstants`: Defines common meteorological constants.
    *   `Thermodynamics`: Contains pure physics functions (e.g., saturation vapor pressure, mixing ratio conversions) operating on NumPy arrays.
    *   `DataBuilder`: Standardizes the creation of `xarray.DataArray` objects, ensuring consistent dimension naming and attribute assignment.
    *   `DiagnosticEngine`: The main engine that detects available input variables and calculates derived diagnostics, adapting to different source datasets (e.g., ERA5, WRF, SFNO). It includes methods for pressure level and surface diagnostics, as well as hydrometeor calculations.
    *   `run_diagnostics`: An entry-point function to execute a standard set of diagnostic calculations.

### `workflow_engine.py`
*   **Purpose**: This is the core orchestration module. It provides a generic, configuration-driven framework to execute complex data processing workflows. It reads YAML configurations, dynamically loads modules, instantiates classes, calls methods, and manages data flow between steps, including handling iterative processes. It is designed to be completely agnostic to the scientific domain (e.g., weather forecasting).
*   **Key Classes/Functions**:
    *   `WorkflowConfigurationError`, `WorkflowExecutionError`: Custom exceptions for workflow management.
    *   `WorkflowEngine`: The central class responsible for parsing configuration, resolving placeholders, managing workflow state, and executing steps including iterative loops. It supports dynamic module/class/method loading and conditional step execution.

### `utils/file_utils.py`
*   **Purpose**: Provides common utility functions for the entire `src/op` module, primarily for logging.
*   **Key Functions**:
    *   `get_logger(name)`: Returns a configured Python logger instance, ensuring consistent logging across the application.

## How it Works Together

The `WorkflowEngine` is the entry point for executing defined operational workflows. A YAML configuration file (e.g., `config/opflows/workflows.yaml`) describes the sequence of steps. Each step specifies a module, class, and method (or action) to execute, along with arguments.

The `WorkflowEngine` dynamically loads the necessary components from `era5_reanalysis.py`, `global_operators.py`, `regridders.py`, and `diagnostics.py` (and potentially other modules). It passes data between steps using a shared `workflow_state` dictionary and handles object instantiation and method calls as directed by the configuration.

This architecture allows for flexible, extensible, and declarative workflow definition without modifying Python code for new sequences or slight variations in data processing.
