# DLAMP.data Gemini Agent Context

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. 
This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

## Project Overview

This project provides a suite of tools for pre-processing and post-processing data for weather forecasting models, specifically the DLAMP.tw and SFNO models. The core functionalities are:

*   **Data Downloading:** Downloading data from the Climate Data Store (CDS).
*   **Regridding:** Regridding the data to a target domain using horizontal and vertical interpolation.
*   **Diagnostics:** Calculating a wide range of diagnostic variables.

The entire workflow is configuration-driven, using YAML files to control the processing steps.

## Building and Running

### Installation

The project uses `micromamba` for environment management and `pip` for installing Python packages.

1.  **Create the conda environment:**
    ```bash
    micromamba env create -n [envname] -c conda-forge python=3.11 conda python-cdo python-eccodes
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    or for a more comprehensive list of dependencies, including those for development and specific models like SFNO:
    ```bash
    pip install -r pyproject.toml
    ```

### Running the Workflow

There are two main workflows, one for ERA5 data and another for SFNO model data.

*   **ERA5 Data Processing:**
    ```bash
    python DLAMPreproc.py
    ```
    The workflow is controlled by `config/era5.yaml`.

*   **SFNO Model Data Processing:**
    ```bash
    python SFNOPreproc.py
    ```
    The workflow is controlled by `config/sfno.yaml`.

## Development Conventions

### Configuration-Driven Workflow

The entire data processing pipeline is controlled by YAML configuration files located in the `config/` directory. This allows for easy modification of parameters such as date ranges, geographical areas, and diagnostic variables without changing the source code.

### Modular Architecture

The codebase is organized into three main components:

1.  **`src/preproc`:** Handles the preprocessing steps, including downloading and regridding.
2.  **`src/registry`:** Manages the calculation of diagnostic variables.
3.  **`config`:** Contains the YAML configuration files.

### Extensible Diagnostics

The diagnostic calculation system is designed to be easily extensible. To add a new diagnostic variable:

1.  **Define the variable in the YAML configuration:** Add a new entry in the `registry.varname` section of the relevant YAML file (e.g., `config/era5.yaml`), specifying the required input variables and the name of the function to calculate it.

2.  **Implement the calculation function:** Add a new Python function to `src/registry/diagnostic_functions.py` (or the `updated.py` file, which seems to be a newer version). This function should take the `source_dataset` and an `xarray.Dataset` as input and return an `xarray.DataArray`.

The system automatically handles the order of execution based on the dependencies defined in the `requires` list for each variable, using a topological sort.

### Code Quality Tools

The project utilizes `ruff` for linting and `radon` for code complexity analysis. These tools are specified in `pyproject.toml` and can be run using the following commands:

*   **Ruff (Linting and Formatting):**
    ```bash
    ruff check .
    ruff format .
    ```

*   **Radon (Code Complexity Analysis):**
    ```bash
    radon cc .
    radon raw .
    radon mi .
    ```

## Active Technologies
- Python 3.11 + micromamba, pip, python-cdo, python-eccodes, cfgrib, dask, xarray (001-module-workflow-spec)
- NetCDF4 files (001-module-workflow-spec)

## Recent Changes
- 001-module-workflow-spec: Added Python 3.11 + micromamba, pip, python-cdo, python-eccodes, cfgrib, dask, xarray
