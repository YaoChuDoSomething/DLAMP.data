# `src/op` Module Structure

This directory (`src/op`) contains the core operational components of the DLAMP data processing pipeline, refactored to adhere strictly to the Single Responsibility Principle (SRP) and designed for a configuration-driven workflow. Each Python file encapsulates a distinct set of functionalities, making the system modular, testable, and maintainable.

## Module Breakdown

### `era5_reanalysis.py`
*   **Purpose**: Manages the retrieval and initial processing of ERA5 reanalysis data from the CDS API. It handles downloading GRIB files, merging them, converting to NetCDF, and basic data validation.
*   **Key Classes/Functions**:
    *   `DataPipelineError`, `CDSRetrievalError`, `ProcessingError`, `ValidationError`: Custom exceptions for pipeline stages.
    *   `CDSRetriever`: Handles direct interaction with the CDS API for data download.
    *   `GribProcessor`: Manages GRIB file operations, including merging multiple GRIBs into a single NetCDF and cleaning up temporary files. Relies on `cdo`.
    *   `DataValidator`: Performs basic checks on the generated NetCDF files to ensure data integrity and presence of required variables/coordinates.
    *   `ERA5DataManager`: Orchestrates the entire ERA5 data preparation process, ensuring both surface and pressure level data are correctly downloaded, merged, and validated into a single NetCDF file for model input.

### `global_operators.py`
*   **Purpose**: Contains functionalities related to global model operations, specifically designed for models like SFNO, supporting both one-way downscaling and two-way coupling workflows. It focuses on data loading, variable standardization, model inference, feedback coupling, and data export.
*   **Key Classes/Functions**:
    *   `GlobalOperatorsError`: Custom exception for errors in global operations.
    *   `VariableMapper`: Standardizes variable names and dimensions for model compatibility, decoupled from file I/O.
    *   `ModelLoader`: Handles loading model weights (e.g., SFNO) and raw NetCDF file I/O.
    *   `InferenceEngine`: Executes model inference in a generator-based fashion, allowing external control for step-by-step processing, crucial for coupled systems.
    *   `FeedbackCoupler`: Implements logic for blending regional model feedback into global fields, supporting two-way coupling.
    *   `DataExporter`: Manages data output, including extracting specific regions for downscaling and saving NetCDF files.

### `regridders.py`
*   **Purpose**: Provides a comprehensive pipeline for data regridding and transformation, essential for preparing global model outputs for regional models or vice-versa. It covers variable standardization, temporal upsampling, vertical coordinate transformation, and optimized spatial regridding.
*   **Key Classes/Functions**:
    *   `PreprocessingError`: Base exception for regridding errors.
    *   `DataStandardizer`: Standardizes variable names, units, and dimensions.
    *   `TimeUpsampler`: Handles temporal interpolation (e.g., 6-hourly to 1-hourly data).
    *   `VerticalTransformer`: Transforms vertical coordinates, such as calculating log-pressure height.
    *   `SpatialRegridder`: Optimizes spatial interpolation from global to regional grids by pre-slicing data with a buffer, then interpolating to a target grid (supporting 2D lat/lon coordinates).
    *   `DataPreparationPipeline`: Orchestrates the sequence of standardization, temporal, vertical, and spatial transformations.

### `diagnostics.py`
*   **Purpose**: Calculates a wide range of diagnostic meteorological variables from raw model outputs or reanalysis data. It separates physical constants, thermodynamic calculations, data building, and the diagnostic engine itself.
*   **Key Classes/Functions**:
    *   `MetConstants`: Defines common meteorological constants.
    *   `Thermodynamics`: Contains pure physics functions (e.g., saturation vapor pressure, mixing ratio conversions) operating on NumPy arrays.
    *   `DataBuilder`: Standardizes the creation of `xarray.DataArray` objects, ensuring consistent dimension naming and attribute assignment.
    *   `DiagnosticEngine`: The main engine that detects available input variables and calculates derived diagnostics, adapting to different source datasets (e.g., ERA5, WRF, SFNO). It includes methods for pressure level and surface diagnostics, as well as hydrometeor calculations.
    *   `run_diagnostics`: An entry-point function to execute a standard set of diagnostic calculations.

### `workflow_engine.py`
*   **Purpose**: This is the core orchestration module. It provides a generic, configuration-driven framework to execute complex data processing workflows. It reads YAML configurations, dynamically loads modules, instantiates classes, calls methods, and manages data flow between steps, including handling iterative processes. It is designed to be completely agnostic to the scientific domain (e.g., weather forecasting).
*   **Key Classes/Functions**:
    *   `WorkflowConfigurationError`, `WorkflowExecutionError`: Custom exceptions for workflow management.
    *   `WorkflowEngine`: The central class responsible for parsing configuration, resolving placeholders, managing workflow state, and executing steps including iterative loops. It supports dynamic module/class/method loading and conditional step execution.

### `utils/file_utils.py`
*   **Purpose**: Provides common utility functions for the entire `src/op` module, primarily for logging.
*   **Key Functions**:
    *   `get_logger(name)`: Returns a configured Python logger instance, ensuring consistent logging across the application.

## How it Works Together

The `WorkflowEngine` is the entry point for executing defined operational workflows. A YAML configuration file (e.g., `config/opflows/workflows.yaml`) describes the sequence of steps. Each step specifies a module, class, and method (or action) to execute, along with arguments.

The `WorkflowEngine` dynamically loads the necessary components from `era5_reanalysis.py`, `global_operators.py`, `regridders.py`, and `diagnostics.py` (and potentially other modules). It passes data between steps using a shared `workflow_state` dictionary and handles object instantiation and method calls as directed by the configuration.

This architecture allows for flexible, extensible, and declarative workflow definition without modifying Python code for new sequences or slight variations in data processing.

# `src/op` 模組結構 (繁體中文 - 臺灣)

此目錄 (`src/op`) 包含 DLAMP 資料處理管線的核心操作組件，經過重構以嚴格遵循單一職責原則 (SRP)，並設計為配置驅動的工作流程。每個 Python 檔案都封裝了一組獨特的功能，使系統更具模組化、可測試性及可維護性。

## 模組細分

### `era5_reanalysis.py`
*   **目的**: 管理從 CDS API 檢索和初始處理 ERA5 再分析資料。它負責下載 GRIB 檔案、合併它們、轉換為 NetCDF 格式，以及進行基本的資料驗證。
*   **主要類別/函數**:
    *   `DataPipelineError`, `CDSRetrievalError`, `ProcessingError`, `ValidationError`: 用於管線階段的自定義例外。
    *   `CDSRetriever`: 處理與 CDS API 的直接互動，用於資料下載。
    *   `GribProcessor`: 管理 GRIB 檔案操作，包括將多個 GRIB 檔案合併為單一 NetCDF，以及清理臨時檔案。依賴 `cdo` 工具。
    *   `DataValidator`: 對生成的 NetCDF 檔案執行基本檢查，以確保資料完整性以及所需變數/座標的存在。
    *   `ERA5DataManager`: 協調整個 ERA5 資料準備過程，確保表面和壓力層資料都正確下載、合併和驗證為單一 NetCDF 檔案，以供模型輸入。

### `global_operators.py`
*   **目的**: 包含與全球模型操作相關的功能，專為 SFNO 等模型設計，支援單向降尺度和雙向耦合工作流程。它專注於資料載入、變數標準化、模型推論、回饋耦合和資料匯出。
*   **主要類別/函數**:
    *   `GlobalOperatorsError`: 全局操作中錯誤的自定義例外。
    *   `VariableMapper`: 標準化變數名稱和維度，以實現模型兼容性，與檔案 I/O 解耦。
    *   `ModelLoader`: 處理模型權重 (例如 SFNO) 和原始 NetCDF 檔案 I/O 的載入。
    *   `InferenceEngine`: 以基於生成器的方式執行模型推論，允許外部控制進行逐步處理，這對於耦合系統至關重要。
    *   `FeedbackCoupler`: 實現將區域模型回饋混合到全局場的邏輯，支持雙向耦合。
    *   `DataExporter`: 管理資料匯出，包括提取特定區域進行降尺度和保存 NetCDF 檔案。

### `regridders.py`
*   **目的**: 提供用於資料重網格和轉換的綜合管線，對於準備全球模型輸出以供區域模型使用或反之，都至關重要。它涵蓋變數標準化、時間向上採樣、垂直座標轉換和優化的空間重網格。
*   **主要類別/函數**:
    *   `PreprocessingError`: 重網格錯誤的基礎例外。
    *   `DataStandardizer`: 標準化變數名稱、單位和維度。
    *   `TimeUpsampler`: 處理時間內插 (例如，6 小時到 1 小時的資料)。
    *   `VerticalTransformer`: 轉換垂直座標，例如計算對數壓力高度。
    *   `SpatialRegridder`: 通過帶緩衝區的預切片資料，然後內插到目標網格 (支援 2D 經緯度座標) 來優化從全局到區域網格的空間內插。
    *   `DataPreparationPipeline`: 協調標準化、時間、垂直和空間轉換的順序。

### `diagnostics.py`
*   **目的**: 從原始模型輸出或再分析資料計算各種診斷氣象變數。它將物理常數、熱力學計算、資料建構和診斷引擎本身分開。
*   **主要類別/函數**:
    *   `MetConstants`: 定義常見的氣象常數。
    *   `Thermodynamics`: 包含作用於 NumPy 陣列的純物理函數 (例如，飽和蒸氣壓、混合比轉換)。
    *   `DataBuilder`: 標準化 `xarray.DataArray` 物件的創建，確保一致的維度命名和屬性分配。
    *   `DiagnosticEngine`: 偵測可用輸入變數並計算導出診斷的主要引擎，適應不同的來源資料集 (例如，ERA5、WRF、SFNO)。它包括壓力層和表面診斷，以及水凝物計算的方法。
    *   `run_diagnostics`: 執行一組標準診斷計算的入口函數。

### `workflow_engine.py`
*   **目的**: 這是核心的協調模組。它提供了一個通用的、配置驅動的框架來執行複雜的資料處理工作流程。它讀取 YAML 配置，動態載入模組、實例化類別、調用方法，並管理步驟之間的資料流，包括處理迭代過程。它被設計為完全不關心科學領域 (例如，天氣預報)。
*   **主要類別/函數**:
    *   `WorkflowConfigurationError`, `WorkflowExecutionError`: 用於工作流程管理的自定義例外。
    *   `WorkflowEngine`: 負責解析配置、解析佔位符、管理工作流程狀態以及執行包括迭代循環在內的步驟的中心類別。它支援動態模組/類別/方法載入和條件式步驟執行。

### `utils/file_utils.py`
*   **目的**: 為整個 `src/op` 模組提供通用的實用函數，主要用於日誌記錄。
*   **主要函數**:
    *   `get_logger(name)`: 返回一個已配置的 Python 日誌記錄器實例，確保整個應用程式的日誌記錄一致。

## 協同運作方式

`WorkflowEngine` 是執行定義的操作工作流程的入口點。YAML 配置檔案 (例如 `config/opflows/workflows.yaml`) 描述了步驟序列。每個步驟都指定了要執行的模組、類別和方法 (或動作)，以及參數。

`WorkflowEngine` 動態載入 `era5_reanalysis.py`、`global_operators.py`、`regridders.py` 和 `diagnostics.py` (以及可能其他模組) 中所需的組件。它使用共享的 `workflow_state` 字典在步驟之間傳遞資料，並根據配置的指示處理物件實例化和方法調用。

這種架構允許靈活、可擴展和聲明式的工作流程定義，而無需修改 Python 程式碼即可實現新的序列或資料處理的微小變化。