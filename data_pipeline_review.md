### 專案資料管線架構分析

此管線的核心目標是**將全球、低時間解析度（6小時）的氣象再分析資料，轉換為區域性、高時間解析度（1小時）、符合 RWRF 模式輸入格式的資料**。

整體來看，這是一個典型的 **ETL (Extract, Transform, Load)** 流程。

#### 1. 識別資料來源 (Data Sources)

*   **主要資料來源**:
    *   **來源**: 哥白尼氣候數據商店 (Copernicus Climate Data Store, CDS)。
    *   **資料集**: ERA5 (ECMWF Reanalysis v5)，這是全球大氣的再分析資料。
    *   **存取方式**: 透過 `cdsapi` Python 客戶端程式庫，由 `src/preproc/cds_downloader.py` 腳本進行自動化下載。
    *   **資料內容**: 分為兩個部分下載：
        1.  **壓力層資料 (`reanalysis-era5-pressure-levels`)**: 包含多個氣壓層上的三維大氣變數（如溫度、風、濕度）。
        2.  **單一層資料 (`reanalysis-era5-single-levels`)**: 包含地表或單一大氣層的變數（如地面氣壓、2公尺溫度、10公尺風）。

*   **次要資料來源**:
    *   **來源**: 專案本地的靜態資產檔案。
    *   **檔案**: `assets/target.nc`。
    *   **用途**: 這個 NetCDF 檔案定義了目標區域網格的地理資訊（經緯度 `XLONG`, `XLAT`、地形高度 `HGT` 等）。這是**水平內插**步驟的目標網格。

#### 2. 追蹤 ETL/ELT 流程

整個流程由 `dlamp_prep.py` 腳本進行調度，依序執行各個步驟：

1.  **`[E]` Extract - 資料提取**:
    *   **執行者**: `src/preproc/cds_downloader.py`。
    *   **流程**:
        *   腳本根據 `config/era5.yaml` 中定義的時間範圍 (`start`, `end`) 和時間間隔 (`base_step_hours: 6`)，產生一個 6 小時為間隔的時間序列。
        *   對於每一個時間點，它會向 CDS API 發送兩個請求，分別下載壓力層和單一層的 GRIB 格式資料。
        *   下載的原始資料暫時存放在 `data/grib/` 目錄下。

2.  **`[T]` Transform - 初步轉換**:
    *   **執行者**: `cdo` (Climate Data Operators)，在 `cds_downloader.py` 中被呼叫。
    *   **流程**:
        *   `cdo` 將同一個時間點的壓力層和單一層 GRIB 檔案**合併**成單一檔案。
        *   同時進行**格式轉換** (GRIB -> NetCDF4) 和**維度調整** (`invertlat`，反轉緯度順序，這是為了符合某些模式的慣例)。
        *   **`[L]` Load**: 轉換後的 NetCDF 檔案被載入（儲存）到 `data/ncdb/` 目錄，檔名格式為 `e5dlamp_YYYYMMDD_HHMI.nc`。
        *   原始的 GRIB 檔案會被刪除以節省空間。

3.  **`[T]` Transform - 核心處理**:
    *   **執行者**: `src/preproc/dlamp_regridder.py`。
    *   **流程**:
        *   **(a) 時間內插 (Temporal Interpolation)**: 腳本根據 `config/era5.yaml` 中 `regrid.output_step_hours` 的設定（例如 1 小時）產生一個高解析度的輸出時間序列。對於每一個輸出的時間點，它會找到前後最接近的兩個 6 小時 NetCDF 檔案，並對所有變數進行**線性內插**。
        *   **(b) 水平內插 (Horizontal Interpolation)**: 將時間內插後的（全球網格）資料，透過 `scipy.griddata` 進行空間上的內插，將其從 ERA5 的全球網格重新網格化 (regrid) 到 `assets/target.nc` 所定義的目標區域網格上。
        *   **(c) 診斷變數計算 (Diagnostic Calculation)**: 根據 `config/era5.yaml` 中 `registry` 區塊的定義，使用 `src/registry/diagnostic_functions.py` 中的函式計算衍生的物理量（例如從露點溫度計算相對濕度）。這個設計具有很好的模組化和擴充性。

4.  **`[L]` Load - 最終載入**:
    *   **執行者**: `src/preproc/dlamp_regridder.py`。
    *   **流程**:
        *   將經過所有轉換步驟（時間內插、水平內插、診斷計算）後的最終資料，以 NetCDF 格式儲存到 `data/ncdb/` 目錄。
        *   最終檔案的命名帶有 `_diag_` 後綴，表示這是包含診斷變數的最終產出。

#### 3. 檢查資料儲存格式

*   **GRIB**: 這是從 CDS 下載的初始格式。它是一種高效的氣象資料壓縮格式，但通用性不如 NetCDF。在此管線中，它只是暫存格式，處理完畢後即被刪除。
*   **NetCDF (`.nc`)**: **這是整個管線中最重要的資料格式**。從初步轉換到最終輸出，都使用 NetCDF。這是一個絕佳的選擇，因為：
    *   **自我描述 (Self-Describing)**: 檔案內部包含維度、座標、變數單位等元數據 (metadata)。
    *   **可攜性高**: 跨平台、跨語言支援度極高。
    *   **生態系完整**: 在氣象和氣候科學領域，有大量工具（如 `xarray`, `cdo`, `nco`）可以高效地處理 NetCDF 檔案。
*   **YAML (`.yaml`)**: 用於儲存所有流程的設定。將設定與程式碼分離，使得管線非常靈活且易於調整。

#### 4. 監控效能瓶頸

根據架構分析，以下是潛在的效能瓶頸：

1.  **資料下載**:
    *   **瓶頸**: I/O 密集型操作，且受限於外部 CDS 伺服器的回應速度和網路頻寬。目前流程是**序列執行**的，即一個時間點下載完成後才開始下一個。
    *   **優化建議**: 下載不同時間點的資料是完全獨立的任務，非常適合**平行化**。可以修改 `dlamp_prep.py`，使用 `multiprocessing` 或 `concurrent.futures` 等套件，同時對多個時間點發起下載請求，能大幅縮短總下載時間。

2.  **水平內插**:
    *   **瓶頸**: CPU 密集型操作。`scipy.griddata` 是一個通用插值工具，但在處理大規模網格資料時，其效能可能不是最佳的。對於每一個變數的每一個壓力層，都需要進行一次完整的空間內插。
    *   **優化建議**: 考慮使用專為地球科學資料設計的 regridding 工具，例如 **`xESMF`**。`xESMF` 底層使用 Fortran 編譯的 ESMF 程式庫，專門處理球座標上的守恆或非守層內插，其效能通常遠高於 `scipy`。

3.  **序列處理流程**:
    *   **瓶頸**: `dlamp_regridder.py` 的主迴圈 `main_process` 也是**序列執行**的，一次只處理一個輸出的時間點。
    *   **優化建議**: 與下載步驟類似，每個輸出時間點的處理（內插、診斷）也是獨立的。可以將 `main_process` 迴圈進行**平行化**，將每個 `curr_time` 的處理分配給不同的 CPU 核心。使用 `Dask` 框架或 `multiprocessing` 都能有效提升整體處理效率。

4.  **I/O 操作**:
    *   **瓶頸**: 在 `main_process` 迴圈中，每個時間點都需要讀取兩個 NetCDF 檔案並寫入一個最終檔案。當資料量非常大或在網路檔案系統 (NFS) 上操作時，頻繁的 I/O 可能會成為瓶頸。
    *   **優化建議**: 目前的設計在 I/O 方面是合理的。若要進一步優化，可以考慮一次性讀取更多資料到記憶體中，但這會增加對 RAM 的要求。對於目前的架構，優先平行化 CPU 密集型任務是更好的選擇。

---