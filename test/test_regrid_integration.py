#!/usr/bin/env python
"""
Integration test for the complete regrid pipeline
測試完整的 regrid + 時間內插流程
"""

import os
import sys

# import shutil
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.core.pipeline import Pipeline
# from src.core.context import Context
import yaml


def test_regrid_integration():
    """整合測試：檢查完整流程是否正常運作"""

    print("=== Regrid Integration Test ===\n")

    # 1. 檢查設定檔
    config_path = "config/era5.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. 檢查關鍵設定
    regrid_cfg = config.get("regrid", {})
    print("Regrid Configuration:")
    print(f"  - method: {regrid_cfg.get('method')}")
    print(f"  - output_step_hours: {regrid_cfg.get('output_step_hours')}")
    print(f"  - enable_temporal_interp: {regrid_cfg.get('enable_temporal_interp')}")
    print(f"  - crop_buffer_degrees: {regrid_cfg.get('crop_buffer_degrees')}")
    print()

    # 3. 檢查輸入檔案是否存在
    share_cfg = config.get("share", {})
    io_cfg = share_cfg.get("io_control", {})
    base_dir = io_cfg.get("base_dir", share_cfg.get("data_path", "."))
    netcdf_subdir = io_cfg.get("netcdf_subdir", "data/ncdb")
    netcdf_dir = os.path.join(str(base_dir), str(netcdf_subdir))
    prefix = io_cfg.get("prefix", {})

    input_files = list(Path(netcdf_dir).glob(f"{prefix['output']}_*.nc"))
    print(f"Found {len(input_files)} input files (e5dlamp_*.nc)")

    if len(input_files) == 0:
        print("⚠️  No input files found. Run download + grib2nc first.")
        return False

    # 4. 檢查 target grid
    target_nc = regrid_cfg.get("target_nc")
    if not os.path.exists(target_nc):
        print(f"❌ Target grid file not found: {target_nc}")
        return False

    print(f"✅ Target grid: {target_nc}\n")

    # 5. 預期輸出檔案數量
    from datetime import datetime

    time_cfg = config["share"]["time_control"]
    start_t = datetime.strptime(time_cfg["start"], time_cfg["format"])
    end_t = datetime.strptime(time_cfg["end"], time_cfg["format"])
    base_step_hours = time_cfg["base_step_hours"]
    output_step_hours = regrid_cfg["output_step_hours"]

    total_boundary = int((end_t - start_t).total_seconds() / 3600 / base_step_hours) + 1
    total_output = int((end_t - start_t).total_seconds() / 3600 / output_step_hours) + 1

    print("Expected outputs:")
    print(f"  - Boundary files (6hr): {total_boundary}")
    print(f"  - Total files (1hr): {total_output}")
    print(f"  - Interpolated files: {total_output - total_boundary}\n")

    # 6. 計算已完成的檔案
    regrid_files = list(Path(netcdf_dir).glob(f"{prefix['regrid']}_*.nc"))
    print(f"✅ Already processed: {len(regrid_files)}/{total_output} regrid files\n")

    # 7. 檢查裁切效果（讀取一個檔案）
    if regrid_files:
        import xarray as xr

        sample_file = regrid_files[0]
        with xr.open_dataset(sample_file) as ds:
            print(f"Sample output file: {sample_file.name}")
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Variables: {list(ds.data_vars)[:5]}...")
            print()

    print("=== Test Summary ===")
    print("✅ Configuration valid")
    print(f"✅ Input files: {len(input_files)}")
    print(f"✅ Processed: {len(regrid_files)}/{total_output}")

    if len(regrid_files) == total_output:
        print("\n🎉 All files processed successfully!")
        return True
    else:
        print(
            f"\n⚠️  Missing {total_output - len(regrid_files)} files. Run pipeline to complete."
        )
        return False


if __name__ == "__main__":
    success = test_regrid_integration()
    sys.exit(0 if success else 1)
