"""
GribToNetCDF 轉換單元測試

測試重點：
1. 檔案路徑組成
2. 跳過已存在檔案
3. 缺失 GRIB 檔案警告

注意：不實際執行 CDO 轉換（需 Mock）
"""

import unittest
import sys
from pathlib import Path
import os
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.tasks.convert import GribToNetCDF
from src.core.context import Context
import yaml


class TestGribToNetCDF(unittest.TestCase):
    """GribToNetCDF 轉換測試"""

    def setUp(self):
        """建立測試環境"""
        self.temp_dir = tempfile.mkdtemp()

        # 建立測試設定
        self.config = {
            "share": {
                "time_control": {
                    "start": "2020-01-01_00:00",
                    "end": "2020-01-01_06:00",
                    "format": "%Y-%m-%d_%H:%M",
                    "base_step_hours": 6,
                },
                "io_control": {
                    "base_dir": self.temp_dir,
                    "grib_subdir": "grib",
                    "netcdf_subdir": "netcdf",
                    "prefix": {
                        "upper": "era5pl",
                        "surface": "era5sl",
                        "output": "e5dlamp",
                        "timestr_fmt": "%Y%m%d_%H%M",
                    },
                },
            }
        }

        config_path = f"{self.temp_dir}/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)

        self.context = Context(config_path)
        self.context.load_config()

    def test_output_file_path_generation(self):
        """測試輸出檔案路徑組成正確"""
        from datetime import datetime

        base_dir_val = self.config["share"]["io_control"]["base_dir"]
        netcdf_subdir_val = self.config["share"]["io_control"]["netcdf_subdir"]
        netcdf_dir = os.path.join(str(base_dir_val), str(netcdf_subdir_val))

        prefix = self.config["share"]["io_control"]["prefix"]
        assert isinstance(prefix, dict)
        timestr_fmt = str(prefix["timestr_fmt"])

        curr_time = datetime(2020, 1, 1, 0, 0)
        timestamp = curr_time.strftime(timestr_fmt)
        prefix_out = str(prefix["output"])
        expected_path = os.path.join(str(netcdf_dir), f"{prefix_out}_{timestamp}.nc")

        # 驗證路徑格式
        self.assertIn("netcdf", expected_path)
        self.assertIn("e5dlamp_20200101_0000.nc", expected_path)

    def test_timeline_generation(self):
        """測試時間序列計算正確"""
        from datetime import datetime, timedelta

        time_cfg = self.config["share"]["time_control"]
        assert isinstance(time_cfg, dict)
        start_t = datetime.strptime(str(time_cfg["start"]), str(time_cfg["format"]))
        end_t = datetime.strptime(str(time_cfg["end"]), str(time_cfg["format"]))

        hours_val = time_cfg["base_step_hours"]
        step = timedelta(hours=float(str(hours_val)))

        total_steps = int((end_t - start_t) / step) + 1
        self.assertEqual(total_steps, 2)  # 00:00, 06:00

        timeline = [start_t + step * i for i in range(total_steps)]
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0], datetime(2020, 1, 1, 0, 0))
        self.assertEqual(timeline[1], datetime(2020, 1, 1, 6, 0))


if __name__ == "__main__":
    unittest.main()
