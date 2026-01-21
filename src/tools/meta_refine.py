#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重新整理 NetCDF 變數名稱與屬性
- 依照 `rename_map` 重新命名變數
- 只保留變數屬性: "description", "standard_name", "單位"
- 時間座標統一為小寫 `time`
"""

import pathlib
import xarray as xr

# -------------------------------------------------
# 設定
# -------------------------------------------------
# 原始檔案路徑（相對於專案根目錄）
SRC_FILE = pathlib.Path("assets/target.nc")
# 輸出檔案路徑
OUT_FILE = pathlib.Path("assets/target_reformatted.nc")

# 變數重新命名對照表 (原變數名稱: 新變數名稱)
rename_map = {
    # 如果變數名稱已經正確，可保持為空
    "Times": "time",
}

# 變數屬性定義 (變數名稱: {屬性名稱: 屬性值})
meta_config = {
    "time": {
        "description": "time",
        "standard_name": "time",
        "units": "day as %Y%m%d.%f",
    },
    "pres_levels": {
        "description": "pressure_levels",
        "standard_name": "",
        "units": "Pa",
    },
    "XLONG": {
        "description": "longitude",
        "standard_name": "longitude",
        "units": "degree east",
    },
    "XLAT": {
        "description": "latitude",
        "standard_name": "latitude",
        "units": "degree north",
    },
    "HGT": {
        "description": "GMTED2010 30-arc-second topography height",
        "standard_name": "terrain_height",
        "units": "m",
    },
    "LANDMASK": {
        "description": "land_sea_mask",
        "standard_name": "land_sea_mask",
        "units": "none",
    },
}


# -------------------------------------------------
# 主流程
# -------------------------------------------------
def main() -> None:
    ds = xr.open_dataset(SRC_FILE)

    # 1. 重新命名變數
    if rename_map:
        ds = ds.rename(rename_map)

    # 2. 簡化與重整變數屬性
    for var in list(ds.data_vars) + list(ds.coords):
        if var in meta_config:
            # 強制設定為要求的屬性
            target_meta = meta_config[var]
            ds[var].attrs = {
                "description": target_meta["description"],
                "standard_name": target_meta["standard_name"],
                "units": target_meta["units"],
            }
            # 如果 standard_name 為空，則移除該項
            if not ds[var].attrs["standard_name"]:
                del ds[var].attrs["standard_name"]

    # 3. 統一時間座標名稱為小寫 `time`
    # 優先處理已有 meta_config 的情況
    if "time" in ds.variables and ds["time"].dims == ("time",):
        pass  # Already correct

    # 檢查其他可能的名稱
    possible_time_names = ["time", "Time", "TIME", "Times"]
    time_coord = None
    for v in list(ds.variables):
        if v in possible_time_names:
            time_coord = v
            break

    if time_coord and time_coord != "time":
        ds = ds.rename({time_coord: "time"})

    # 4. 寫出新檔案
    ds.to_netcdf(OUT_FILE)
    print(f"已完成重新整理，輸出檔案: {OUT_FILE}")


if __name__ == "__main__":
    main()
