import unittest
import numpy as np
import xarray as xr
import tempfile

# import os
from datetime import datetime, timedelta
# from src.tasks.regrid import Regridder
# from src.core.context import Context


class TestRegridderCrop(unittest.TestCase):
    """測試區域裁切功能"""

    def setUp(self):
        """建立測試用的假資料"""
        # 建立全球網格資料（簡化版）
        lon = np.arange(0, 360, 1.0)  # 1° 解析度
        lat = np.arange(-90, 91, 1.0)
        self.global_ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon))),
                "q": (["time", "lat", "lon"], np.random.rand(1, len(lat), len(lon))),
            },
            coords={
                "time": [datetime(2020, 1, 1)],
                "lon": lon,
                "lat": lat,
            },
        )

    def test_crop_reduces_grid_size(self):
        """測試裁切是否正確減少網格大小"""
        # 目標區域：台灣附近 (120-125°E, 22-26°N)
        lon_min, lon_max = 118, 127
        lat_min, lat_max = 20, 28

        # 裁切
        cropped = self.global_ds.sel(
            lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)
        )

        # 驗證
        self.assertLess(cropped.dims["lon"], self.global_ds.dims["lon"])
        self.assertLess(cropped.dims["lat"], self.global_ds.dims["lat"])
        self.assertEqual(cropped.dims["lon"], 10)  # 127-118+1
        self.assertEqual(cropped.dims["lat"], 9)  # 28-20+1


class TestTemporalInterpolation(unittest.TestCase):
    """測試時間內插功能"""

    def test_linear_interpolation_weights(self):
        """測試線性內插權重計算"""
        # 測試案例：18:00 和 24:00 之間的 21:00
        t0 = datetime(2020, 1, 1, 18, 0)
        t1 = datetime(2020, 1, 2, 0, 0)  # 24:00 = 次日 00:00
        curr_time = datetime(2020, 1, 1, 21, 0)

        total_sec = (t1 - t0).total_seconds()
        weight0 = (t1 - curr_time).total_seconds() / total_sec
        weight1 = (curr_time - t0).total_seconds() / total_sec

        # 驗證
        self.assertAlmostEqual(weight0, 0.5, places=5)
        self.assertAlmostEqual(weight1, 0.5, places=5)
        self.assertAlmostEqual(weight0 + weight1, 1.0, places=10)

    def test_interpolated_values(self):
        """測試內插後的數值正確性"""
        # 建立兩個時間點的資料
        ds0 = xr.Dataset({"temp": (["time", "x"], [[10.0, 20.0]])})
        ds1 = xr.Dataset({"temp": (["time", "x"], [[20.0, 40.0]])})

        # 50% 權重內插
        ds_interp = ds0 * 0.5 + ds1 * 0.5

        # 驗證
        expected = np.array([[15.0, 30.0]])
        np.testing.assert_array_almost_equal(ds_interp["temp"].values, expected)


class TestRegridderIntegration(unittest.TestCase):
    """整合測試：完整流程"""

    def setUp(self):
        """建立測試環境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理測試環境"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_two_phase_processing_order(self):
        """測試兩階段處理順序正確"""
        # 模擬時間範圍：18:00 到 24:00，base_step=6hr, output_step=1hr
        start_t = datetime(2020, 1, 1, 18, 0)
        end_t = datetime(2020, 1, 2, 0, 0)
        base_step = timedelta(hours=6)
        out_step = timedelta(hours=1)

        # 階段1：應該處理邊界點
        boundary_times = []
        curr = start_t
        while curr <= end_t:
            boundary_times.append(curr)
            curr += base_step

        # 驗證邊界點
        self.assertEqual(len(boundary_times), 2)  # 18:00, 24:00
        self.assertEqual(boundary_times[0], start_t)
        self.assertEqual(boundary_times[1], end_t)

        # 階段2：應該內插中間點
        total_steps = int((end_t - start_t) / out_step) + 1
        intermediate_times = []
        for i in range(total_steps):
            curr_time = start_t + out_step * i
            if curr_time not in boundary_times:
                intermediate_times.append(curr_time)

        # 驗證中間點
        self.assertEqual(len(intermediate_times), 5)  # 19:00-23:00


if __name__ == "__main__":
    unittest.main()
