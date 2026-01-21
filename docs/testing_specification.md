# 測試規範說明書

## 1. 測試策略總覽

### 測試目標

- **覆蓋率**: >= 80% 程式碼覆蓋率
- **測試類型**: 單元測試、整合測試、資料驗證測試
- **自動化**: CI/CD 自動執行
- **可重現性**: 固定 seed、合成資料

### 測試範圍

| 模組 | 測試檔案 | 覆蓋範圍 |
| :--- | :--- | :--- |
| **Core** | `test_pipeline.py` | Pipeline 執行、Context 載入 |
| **Tasks** | `test_convert.py`, `test_regrid.py`, `test_diagnostics.py` | 資料處理流程 |
| **Integration** | `test_*_integration.py` | 端到端流程 |
| **Data Quality** | `test_data_quality.py`, `test_nc_format.py` | 資料驗證 |

---

## 2. 測試案例定義

### 2.1 單元測試（Unit Tests）

#### test_pipeline.py - Pipeline 核心邏輯

| 測試案例 | 待測程式碼 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- | :--- |
| `test_task_execution_order` | [pipeline.py:L55-57](file:///data/dwp/bmds/src/core/pipeline.py#L55-L57) | Tasks 按順序執行 | 執行順序錯誤 |
| `test_context_loading` | [pipeline.py:L34](file:///data/dwp/bmds/src/core/pipeline.py#L34) | Config 正確載入 | Config 遺失或錯誤 |
| `test_empty_pipeline` | [pipeline.py:L32-59](file:///data/dwp/bmds/src/core/pipeline.py#L32-L59) | 無 Exception | 報錯 |

**執行方式**：

```bash
python -m unittest test.test_pipeline -v
```

---

#### test_convert.py - GRIB→NetCDF 轉換

| 測試案例 | 待測程式碼 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- | :--- |
| `test_timeline_generation` | [convert.py:L26-30](file:///data/dwp/bmds/src/tasks/convert.py#L26-L30) | 時間序列正確 | 時間點遺漏/錯誤 |
| `test_output_file_path_generation` | [convert.py:L44-46](file:///data/dwp/bmds/src/tasks/convert.py#L44-L46) | 路徑格式正確 | 路徑錯誤 |

---

#### test_regrid.py - 空間內插

| 測試案例 | 待測程式碼 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- | :--- |
| `test_crop_reduces_grid_size` | [regrid.py:L171-177](file:///data/dwp/bmds/src/tasks/regrid.py#L171-L177) | 網格縮小 > 90% | 裁切無效 |
| `test_linear_interpolation_weights` | [regrid.py:L150-153](file:///data/dwp/bmds/src/tasks/regrid.py#L150-L153) | 權重和 = 1.0 | 權重錯誤 |
| `test_two_phase_processing_order` | [regrid.py:L71-127](file:///data/dwp/bmds/src/tasks/regrid.py#L71-L127) | 邊界點→內插點 | 順序錯誤 |

---

#### test_diagnostics.py - 診斷變數計算

| 測試案例 | 待測程式碼 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- | :--- |
| `test_dependency_sorting` | [diagnostic_registry.py](file:///data/dwp/bmds/src/registry/diagnostic_registry.py) | 依賴順序正確 | 環狀依賴或順序錯 |
| `test_missing_dependency_skip` | [diagnostics.py:L84-96](file:///data/dwp/bmds/src/tasks/diagnostics.py#L84-L96) | 跳過缺失變數 | Crash |

---

### 2.2 整合測試（Integration Tests）

#### test_regrid_integration.py

| 測試案例 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- |
| 設定檔有效性 | YAML 可解析，關鍵欄位存在 | 解析失敗 |
| 輸入檔案檢查 | 找到至少 1 個輸入檔 | 無輸入檔 |
| 預期輸出數量計算 | 公式正確（邊界+內插） | 計算錯誤 |
| 裁切效果驗證 | 格點減少 > 90% | 無裁切效果 |

**執行方式**：

```bash
python test/test_regrid_integration.py
```

---

### 2.3 資料驗證測試（Data Validation Tests）

#### test_data_quality.py

| 測試案例 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- |
| `test_no_nan_in_critical_vars` | NaN < 1% | NaN > 1%（除 fill_value） |
| `test_value_range_validation` | 溫度 200-330K，壓力 > 0 | 超出合理範圍 |
| `test_spatial_continuity` | 相鄰格點差異 < 閾值 | 出現異常跳變 |

#### test_nc_format.py

| 測試案例 | 通過標準 | 失敗標準 |
| :--- | :--- | :--- |
| `test_cf_conventions_compliance` | 符合 CF-1.8 | 不符 CF 標準 |
| `test_dimension_consistency` | 所有變數維度名稱一致 | 維度名稱衝突 |
| `test_metadata_completeness` | units, long_name 存在 | 缺少必要屬性 |

---

## 3. 通過/失敗標準詳細定義

### 功能正確性

- **通過**：輸出符合預期值（浮點數誤差 < 1e-5，整數完全匹配）
- **失敗**：輸出錯誤、Exception、靜默失敗

### 邊界條件

- **通過**：正確處理空輸入、極值（0, Inf, -Inf）、邊界索引
- **失敗**：Crash、無限迴圈、記憶體溢出

### 資料完整性

- **通過**：
  - NaN < 1%（除 fill_value 外）
  - 數值範圍：溫度 200-330K，壓力 > 0，風速 < 100 m/s
  - 時間座標單調遞增
- **失敗**：
  - NaN > 1%
  - 數值超出物理合理範圍
  - 時間亂序

### 格式正確性

- **通過**：
  - CF Conventions 1.8 符合性
  - Metadata 包含 units, long_name, coordinates
  - 維度名稱一致（time, plev, lat, lon）
- **失敗**：
  - 不符 CF 標準
  - 缺少必要屬性
  - 維度名稱不一致

---

## 4. 測試執行指南

### 本地執行

#### 完整測試套件

```bash
# 啟動虛擬環境
source .venv/bin/activate

# 執行所有測試
python -m unittest discover test/ -v

# 執行特定測試模組
python -m unittest test.test_pipeline -v
python -m unittest test.test_regrid -v
```

#### 生成覆蓋率報告

```bash
# 安裝 coverage（若未安裝）
pip install coverage

# 執行測試並收集覆蓋率
coverage run -m unittest discover test/ -v
coverage report -m
coverage html  # 產生 htmlcov/index.html

# 查看結果
open htmlcov/index.html
```

#### Fixture 資料生成

```bash
# 生成測試用資料（首次執行或更新時）
python test/generate_fixtures.py
```

### CI/CD 執行

#### GitHub Actions 配置

參見 `.github/workflows/ci_tests.yml`

#### 自動觸發時機

- Push 至 main/develop 分支
- Pull Request 建立或更新
- 每日夜間執行（排程）

---

## 5. 測試覆蓋率目標

| 模組 | 目標覆蓋率 | 當前狀態 |
| :--- | :--- | :--- |
| `src/core/` | >= 90% | ⏳ 待測量 |
| `src/tasks/` | >= 80% | ⏳ 待測量 |
| `src/registry/` | >= 70% | ⏳ 待測量 |
| **整體** | **>= 80%** | **⏳ 待測量** |

### 測量方式

```bash
coverage run -m unittest discover test/ -v
coverage report -m | grep TOTAL
```

---

## 6. 已知問題與限制

### 跳過的測試

- **CDS API 下載測試**：需要真實 API 或 Mock（已跳過）
- **CDO 轉換測試**：需要 CDO 環境（僅測試路徑組成）

### 維度名稱不一致

- **狀態**：✅ 已解決
- **說明**：已統一將 `Time`, `Times` 命名改為小寫 `time`（符合 CF Conventions）。
- **影響**：解決了 `diagnostic_functions.py` 與 `regrid.py` 之間的維度衝突，原本的整合測試現已可正常運行。

### Fixture 限制

- **網格大小**：10x10（真實 721x1440）
- **壓力層**：5 層（真實 31 層）
- **時間點**：單一時間點（真實數百/數千時間點）

---

## 7. 測試維護指南

### 新增測試案例

1. 識別待測功能
2. 建立測試檔案（`test_<module>.py`）3. 撰寫測試函數（`test_<function_name>`）
3. 更新此文件測試案例表格
4. 執行測試確認通過

### 更新 Fixture 資料

```bash
# 編輯 test/generate_fixtures.py
# 重新生成
python test/generate_fixtures.py
```

### 測試失敗處理流程

1. 查看測試輸出錯誤訊息
2. 確認是測試問題或程式問題
3. 修正並重新執行
4. 更新文件（若需要）

---

## 8. 參考資源

- [xarray Testing Guide](https://docs.xarray.dev/en/stable/contributing.html#test-driven-development)
- [CF Conventions](http://cfconventions.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage.py](https://coverage.readthedocs.io/)
