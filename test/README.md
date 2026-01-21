# 測試執行快速指南

## 快速開始

```bash
# 1. 啟動虛擬環境
source .venv/bin/activate

# 2. 生成測試資料（首次執行）
python test/generate_fixtures.py

# 3. 執行所有測試
python -m unittest discover test/ -v
```

## 測試檔案清單

| 測試類型 | 檔案 | 說明 |
| :--- | :--- | :--- |
| **單元測試** | `test_pipeline.py` | Pipeline 核心邏輯 |
| | `test_convert.py` | GRIB→NetCDF 轉換 |
| | `test_regrid.py` | 空間內插 |
| | `test_diagnostics.py` | 診斷變數計算 |
| **整合測試** | `test_regrid_integration.py` | Regrid 完整流程 |
| **資料驗證** | `test_data_quality.py` | 資料品質檢查 |
| | `test_nc_format.py` | NetCDF 格式驗證 |

## 執行特定測試

```bash
# 單一測試檔案
python -m unittest test.test_pipeline -v

# 單一測試類別
python -m unittest test.test_regrid.TestTemporalInterpolation -v

# 單一測試案例
python -m unittest test.test_regrid.TestTemporalInterpolation.test_linear_interpolation_weights -v
```

## 測試覆蓋率

```bash
# 執行並產生覆蓋率報告
coverage run -m unittest discover test/ -v
coverage report -m
coverage html

# 查看 HTML 報告
open htmlcov/index.html
```

## 詳細說明

參見 [`docs/testing_specification.md`](file:///data/dwp/bmds/docs/testing_specification.md)
