# DLAMP.data
Streamlining Your Data-Driven Workflow:  Pre-processing and Post-processing Utilities for DLAMP.tw Model

## How to install the python environment
Condition 1. Simple Environment Setup for DLAMP.data 
```
micromamba env create -n [envname] -c conda-forge python=3.11 conda python-cdo python-eccodes
pip install -r requirement.txt
```

Condition 2. Environment Setup for DLAMP.tw DLAMP.data
    * Please install DLAMP.tw first and freeze python version in 3.11
    * install hydra-core use extra "--upgrade" after installing the requirement records by pip
    * install onnxruntime according to your CUDA version, please check onnxruntime_official for more details.
```
micromamba env create -n [envname] -c conda-forge python=3.11 conda

git clone https://github.com/NVIDIA/physicsnemo && cd physicsnemo
make install && cd ..

git clone https://github.com/Chia-Tung/DLAMP DLAMP.tw && cd DLAMP.tw
pip install -r requirements.txt && \
pip install hydra-core --upgrade && \
pip install onnxruntime-gpu==1.20.0 && cd .. 

git clone https://github.com/YaoChuDoSomething/DLAMP.data DLAMP.data && cd DLAMP.data && cd DLAMP.data
pip install -r requirement.txt
```
