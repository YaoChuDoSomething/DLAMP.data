"""
Fixture 資料設計策略與生成腳本

目的：建立小型、輕量級的測試用資料，避免依賴真實 GRIB/NetCDF 大檔案

設計原則：
1. 合成資料（Synthetic Data）：使用 numpy/xarray 生成
2. 最小化檔案大小：小網格（10x10）、少變數、單一時間點
3. 涵蓋測試場景：正常、邊界、異常值
4. 可重現性：固定 random seed
"""

import numpy as np
import xarray as xr
from datetime import datetime
import os


class FixtureGenerator:
    """測試 Fixture 資料生成器"""

    def __init__(self, output_dir="test/fixtures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(42)  # 固定 seed 確保可重現

    def generate_era5_like_netcdf(
        self, filename="era5_sample.nc", nlat=10, nlon=10, nlevels=5
    ):
        """
        生成類 ERA5 格式的 NetCDF（轉換後格式）

        維度：time, plev, lat, lon
        變數：t, q, u, v, z（溫度、比濕、風速、位勢高度）
        """
        # 座標
        lat = np.linspace(20, 30, nlat)  # 台灣周邊
        lon = np.linspace(118, 128, nlon)
        plev = np.array([1000, 925, 850, 700, 500])[:nlevels]  # 壓力層（hPa）
        time = [datetime(2020, 1, 1, 0, 0)]

        # 生成合理範圍的假資料
        ds = xr.Dataset(
            {
                # 溫度（K）：200-330K
                "t": (
                    ["time", "plev", "lat", "lon"],
                    np.random.uniform(250, 300, (1, nlevels, nlat, nlon)),
                ),
                # 比濕（kg/kg）：0-0.02
                "q": (
                    ["time", "plev", "lat", "lon"],
                    np.random.uniform(0.001, 0.015, (1, nlevels, nlat, nlon)),
                ),
                # U 風速（m/s）：-50 to 50
                "u": (
                    ["time", "plev", "lat", "lon"],
                    np.random.uniform(-20, 20, (1, nlevels, nlat, nlon)),
                ),
                # V 風速（m/s）：-50 to 50
                "v": (
                    ["time", "plev", "lat", "lon"],
                    np.random.uniform(-20, 20, (1, nlevels, nlat, nlon)),
                ),
                # 位勢高度（m）：500-30000m
                "z": (
                    ["time", "plev", "lat", "lon"],
                    np.random.uniform(1000, 15000, (1, nlevels, nlat, nlon)),
                ),
            },
            coords={
                "time": time,
                "plev": plev,
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "title": "Test Fixture - ERA5-like data",
                "source": "Synthetic data for unit testing",
            },
        )

        output_path = os.path.join(self.output_dir, filename)
        ds.to_netcdf(output_path, format="NETCDF4")
        print(f"✅ Generated: {output_path}")
        return output_path

    def generate_regrid_target(self, filename="target.nc", ny=15, nx=15):
        """
        生成 regrid 目標網格（類似 WRF grid）

        維度：time, south_north, west_east
        變數：XLONG, XLAT, HGT, LANDMASK
        """
        # 目標網格座標（略小於 ERA5 範圍）
        xlat = np.linspace(21, 29, ny)
        xlong = np.linspace(119, 127, nx)
        xlat_2d, xlong_2d = np.meshgrid(xlat, xlong, indexing="ij")

        ds = xr.Dataset(
            {
                "XLONG": (["time", "south_north", "west_east"], [xlong_2d]),
                "XLAT": (["time", "south_north", "west_east"], [xlat_2d]),
                "HGT": (
                    ["time", "south_north", "west_east"],
                    [np.random.uniform(0, 3000, (ny, nx))],
                ),  # 地形高度
                "LANDMASK": (
                    ["time", "south_north", "west_east"],
                    [np.random.choice([0, 1], (ny, nx))],
                ),  # 陸地遮罩
                "pres_levels": (["pres_bottom_top"], [1000, 925, 850, 700, 500]),
            },
            coords={
                "time": [datetime(2020, 1, 1)],
                "south_north": np.arange(ny),
                "west_east": np.arange(nx),
                "pres_bottom_top": [1000, 925, 850, 700, 500],
            },
        )

        output_path = os.path.join(self.output_dir, filename)
        ds.to_netcdf(output_path, format="NETCDF4")
        print(f"✅ Generated: {output_path}")
        return output_path

    def generate_edge_case_data(self):
        """生成邊界/異常測試資料"""
        # 1. 含 NaN 的資料
        ds_nan = xr.Dataset(
            {
                "temp": (
                    ["time", "lat", "lon"],
                    np.array([[[np.nan, 1, 2], [3, 4, np.nan], [5, 6, 7]]]),
                )
            },
            coords={"time": [datetime(2020, 1, 1)], "lat": [0, 1, 2], "lon": [0, 1, 2]},
        )
        ds_nan.to_netcdf(f"{self.output_dir}/data_with_nan.nc")

        # 2. 超出合理範圍的資料
        ds_invalid = xr.Dataset(
            {
                "temp": (["time", "lat"], np.array([[400, -100]]))  # 異常溫度
            },
            coords={"time": [datetime(2020, 1, 1)], "lat": [0, 1]},
        )
        ds_invalid.to_netcdf(f"{self.output_dir}/data_invalid_range.nc")

        print("✅ Generated edge case fixtures")

    def generate_minimal_config(self):
        """生成測試用最小設定檔"""
        config = """# Minimal test configuration
share:
  exp_code: "TEST"
  data_path: "./test/fixtures"
  time_control:
    start: "2020-01-01_00:00"
    end: "2020-01-01_06:00"
    format: "%Y-%m-%d_%H:%M"
    base_step_hours: 6
  io_control:
    base_dir: "./test/fixtures"
    grib_subdir: "grib"
    netcdf_subdir: "ncdb"
    npy_subdir: "npy"
    prefix:
      upper: "test_upper"
      surface: "test_surface"
      regrid: "test_regrid"
      output: "test_output"
      timestr_fmt: "%Y%m%d_%H%M"

regrid:
  method: "bilinear"
  output_step_hours: 1
  target_nc: "./test/fixtures/target.nc"
  target_lat: "XLAT"
  target_lon: "XLONG"
  source_lat: "lat"
  source_lon: "lon"
  levels: [1000, 925, 850, 700, 500]
  adopted_varlist: ["XLONG", "XLAT", "pres_levels", "HGT", "LANDMASK"]
  use_dask: false
  crop_buffer_degrees: 2.0
  enable_temporal_interp: true

registry:
  source_dataset: "TEST"
  varname: {}
"""
        with open(f"{self.output_dir}/test_config.yaml", "w") as f:
            f.write(config)
        print("✅ Generated: test/fixtures/test_config.yaml")


def main():
    """主執行函數：生成所有 fixture 資料"""
    print("=== Generating Test Fixtures ===\n")

    gen = FixtureGenerator()

    # 1. 生成類 ERA5 NetCDF
    gen.generate_era5_like_netcdf("era5_sample_10x10.nc", nlat=10, nlon=10, nlevels=5)

    # 2. 生成目標網格
    gen.generate_regrid_target("target.nc", ny=15, nx=15)

    # 3. 生成邊界測試資料
    gen.generate_edge_case_data()

    # 4. 生成測試設定檔
    gen.generate_minimal_config()

    print("\n=== Fixture Generation Complete ===")
    print("Files created in test/fixtures/")
    print("  - era5_sample_10x10.nc (~5KB)")
    print("  - target.nc (~3KB)")
    print("  - data_with_nan.nc")
    print("  - data_invalid_range.nc")
    print("  - test_config.yaml")


if __name__ == "__main__":
    main()
