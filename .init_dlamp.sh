#!/bin/sh
#


uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.9.0"
uv add earth2studio --extra sfno --extra data

uv add xarray netCDF4 h5netcdf h5py cfgrib cftime eccodes dask # wgrib pydap
uv add notebook jupyter-http-over-ws ipykernel markitdown
uv add metpy cartopy geopandas rasterio ffmpeg matplotlib scipy
uv add importlib-metadata requests termcolor wandb mlflow opencv-python pyyaml pydantic lightning einops hydra-core onnxruntime-gpu

uv add pytest ruff radon black flake8 isort pydocstyle


