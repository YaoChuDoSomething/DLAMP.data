# DLAMP.data

Streamlining Your Data-Driven Workflow:  Pre-processing and Post-processing Utilities for DLAMP.tw Model

## How to install the python environment

Condition 1. Simple Environment Setup for DLAMP.data

```bash
uv init --python 3.12
uv venv [--clear]
source .venv/bin/activate

apt install cdo eccodes ffmpeg
uv sync
uv sync --dev
```

Condition 2. Environment Setup for DLAMP.tw and DLAMP.data [Experimental]

* Please install DLAMP.tw first and freeze python version in 3.11
* install hydra-core use extra "--upgrade" after installing the requirement records by pip
* install onnxruntime according to your CUDA version, please check onnxruntime_official for more details.

```bash
uv venv --python 3.11
source .venv/bin/activate

git clone https://github.com/NVIDIA/physicsnemo && cd physicsnemo
uv pip install . && cd ..

git clone https://github.com/Chia-Tung/DLAMP DLAMP.tw && cd DLAMP.tw
uv pip install -r requirements.txt && \
uv pip install hydra-core --upgrade && \
uv pip install onnxruntime-gpu==1.20.0 && cd ..

git clone https://github.com/YaoChuDoSomething/DLAMP.data DLAMP.data && cd DLAMP.data
uv pip install -r requirement.txt
```
