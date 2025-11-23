#!/bin/bash

micromamba create -n OP -c conda-forge python=3.12 conda
micromamba run -n OP micromamba install -c conda-forge ffmpeg cdo python-cdo eccodes python-eccodes hdf5 netcdf4 h5netcdf h5py
micromamba run -n OP pip install "torch==2.9.0" "torchvision==0.24.0" --index-url "https://download.pytorch.org/whl/cu128"
micromamba run -n OP pip install earth2studio
micromamba run -n OP pip install earth2studio[data]
micromamba run -n OP pip install "makani @ git+https://github.com/NVIDIA/modulus-makani.git@28f38e3e929ed1303476518552c64673bbd6f722"
micromamba run -n OP pip install earth2studio[sfno]

micromamba run -n OP pip install -r requirements.txt

