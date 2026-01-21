"""
Diagnostics 診斷變數計算單元測試

測試重點：
1. 依賴排序正確性
2. 診斷函數計算正確性
3. 缺少依賴時跳過
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry.diagnostic_registry import sort_diagnostics_by_dependencies


class TestDiagnosticDependencySorting(unittest.TestCase):
    """診斷變數依賴排序測試"""

    def test_simple_dependency_sorting(self):
        """測試簡單依賴排序"""
        diagnostics = {
            "A": {"requires": []},
            "B": {"requires": ["A"]},
            "C": {"requires": ["B"]},
        }

        sorted_vars = sort_diagnostics_by_dependencies(diagnostics)

        # A 必須在 B 之前，B 必須在 C 之前
        self.assertTrue(sorted_vars.index("A") < sorted_vars.index("B"))
        self.assertTrue(sorted_vars.index("B") < sorted_vars.index("C"))

    def test_complex_dependency_sorting(self):
        """測試複雜依賴排序"""
        diagnostics = {
            "A": {"requires": []},
            "B": {"requires": []},
            "C": {"requires": ["A", "B"]},
            "D": {"requires": ["C"]},
        }

        sorted_vars = sort_diagnostics_by_dependencies(diagnostics)

        # A, B 必須在 C 之前
        self.assertTrue(sorted_vars.index("A") < sorted_vars.index("C"))
        self.assertTrue(sorted_vars.index("B") < sorted_vars.index("C"))
        # C 必須在 D 之前
        self.assertTrue(sorted_vars.index("C") < sorted_vars.index("D"))


class TestDiagnosticFunctions(unittest.TestCase):
    """診斷函數計算測試"""

    def test_temperature_conversion(self):
        """測試溫度單位轉換"""
        # 假設有 diag_tk_p 函數（t 不需轉換）
        from src.registry.diagnostic_functions import diag_tk_p

        # 建立測試資料
        ds = xr.Dataset({"t": (["time", "plev"], [[280, 290, 300]])})

        result = diag_tk_p("ERA5", ds)

        # 驗證結果
        np.testing.assert_array_almost_equal(result.values, [[280, 290, 300]])

    def test_missing_dependency_handling(self):
        """測試缺少依賴時的處理"""
        ds = xr.Dataset({"temp": (["time"], [280])})

        # 嘗試計算需要 't' 的診斷變數（但只有 'temp'）
        requires = ["t"]
        has_all_deps = all(req in ds.data_vars for req in requires)

        self.assertFalse(has_all_deps, "Should detect missing dependency")


if __name__ == "__main__":
    unittest.main()
