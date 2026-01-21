#!/bin/python
from src.core.context import Context
from src.core.pipeline import Pipeline
from src.tasks.download import ERA5Downloader
from src.tasks.convert import GribToNetCDF
from src.tasks.regrid import Regridder
from src.tasks.diagnostics import Diagnostics


def main():
    # 1. Initialize Context
    config_path = "config/era5.yaml"
    context = Context(config_path=config_path)
    context.load_config()

    # 2. Initialize Pipeline
    pipeline = Pipeline(context)

    # 3. Add Tasks (Strict Order: Download -> Convert -> Regrid -> Diagnose)
    pipeline.add_task(ERA5Downloader())
    pipeline.add_task(GribToNetCDF())
    pipeline.add_task(Regridder())
    pipeline.add_task(Diagnostics())

    # 4. Run
    pipeline.run()


if __name__ == "__main__":
    main()
