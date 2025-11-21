import cdsapi
import xarray as xr
import subprocess
import os
import shutil
from typing import List, Dict, Optional
from pathlib import Path
from .utils.file_utils import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------------------

class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass

class CDSRetrievalError(DataPipelineError):
    """Errors related to CDS API retrieval."""
    pass

class ProcessingError(DataPipelineError):
    """Errors related to file processing (CDO/Merging)."""
    pass

class ValidationError(DataPipelineError):
    """Errors related to data validation."""
    pass

# ------------------------------------------------------------------------------
# 1. API Retriever (負責與 CDS 溝通)
# ------------------------------------------------------------------------------

class CDSRetriever:
    """
    負責執行 CDS API 的請求。
    職責：單純的下載，不涉及檔案合併或格式轉換。
    """
    def __init__(self):
        try:
            self.client = cdsapi.Client()
        except Exception as e:
            raise CDSRetrievalError(f"Failed to initialize CDS API client. Check .cdsapirc file. Error: {e}")

    def retrieve(self, dataset_name: str, request_params: Dict, output_path: str) -> str:
        """
        執行單次下載請求。
        """
        logger.info(f"CDSRetriever: Requesting {dataset_name} -> {output_path}")
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.client.retrieve(dataset_name, request_params, output_path)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise CDSRetrievalError(f"Download finished but file is missing or empty: {output_path}")
            
            logger.info(f"CDSRetriever: Download successful: {output_path}")
            return output_path
            
        except Exception as e:
            raise CDSRetrievalError(f"CDS API error for {dataset_name}: {e}")

# ------------------------------------------------------------------------------
# 2. GRIB Processor (負責檔案操作與 CDO 呼叫)
# ------------------------------------------------------------------------------

class GribProcessor:
    """
    負責處理 GRIB 檔案的操作，包含合併與轉檔 (GRIB -> NetCDF)。
    依賴：系統需安裝 CDO (Climate Data Operators)。
    """
    def __init__(self):
        if not shutil.which("cdo"):
            raise ProcessingError("CDO command not found in PATH. Please install cdo.")

    def merge_and_convert(self, input_files: List[str], output_nc_path: str) -> str:
        """
        將多個 GRIB 檔案合併並轉換為單一 NetCDF 檔案。
        這對於將 Surface 和 Pressure Level 資料合併為模型輸入至關重要。
        """
        if not input_files:
            raise ProcessingError("No input files provided for merging.")
        
        # 檢查輸入檔案是否存在
        for f in input_files:
            if not os.path.exists(f):
                raise ProcessingError(f"Input file not found: {f}")

        logger.info(f"GribProcessor: Merging {len(input_files)} files into {output_nc_path}...")
        
        # 建構 CDO 指令: cdo -f nc4 merge file1.grib file2.grib output.nc
        # -f nc4 強制輸出為 NetCDF4 (支援壓縮)
        cmd = ["cdo", "-f", "nc4", "merge"] + input_files + [output_nc_path]
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info(f"GribProcessor: Merge successful.")
            return output_nc_path
            
        except subprocess.CalledProcessError as e:
            err_msg = f"CDO merge failed.\nCommand: {' '.join(cmd)}\nStderr: {e.stderr}"
            logger.error(err_msg)
            raise ProcessingError(err_msg)
        except Exception as e:
            raise ProcessingError(f"Unexpected error during merging: {e}")

    def cleanup(self, files_to_remove: List[str]):
        """清理暫存檔案"""
        for f in files_to_remove:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logger.debug(f"Removed temp file: {f}")
            except OSError as e:
                logger.warning(f"Failed to remove temp file {f}: {e}")

# ------------------------------------------------------------------------------
# 3. Data Validator (負責資料完整性檢查)
# ------------------------------------------------------------------------------

class DataValidator:
    """
    負責驗證產出的 NetCDF 檔案是否符合模型輸入需求。
    """
    def validate_nc(self, file_path: str, required_vars: Optional[List[str]] = None) -> xr.Dataset:
        logger.info(f"DataValidator: Checking {file_path}...")
        if not os.path.exists(file_path):
            raise ValidationError(f"File not found: {file_path}")

        try:
            # 使用 xarray 開啟檢查，但不載入數據進記憶體
            ds = xr.open_dataset(file_path)
            
            if required_vars:
                file_vars = set(ds.data_vars)
                missing = set(required_vars) - file_vars
                if missing:
                    raise ValidationError(f"Missing required variables in {file_path}: {missing}")
            
            # 簡單檢查維度
            if 'time' not in ds.coords:
                raise ValidationError("Dataset missing 'time' coordinate.")
                
            logger.info("DataValidator: Validation passed.")
            return ds
            
        except Exception as e:
            raise ValidationError(f"Invalid NetCDF file {file_path}: {e}")

# ------------------------------------------------------------------------------
# 4. Orchestrator (協調者：整合下載與合併流程)
# ------------------------------------------------------------------------------

class ERA5DataManager:
    """
    高層介面：協調下載與處理流程。
    確保地面資料 (Single Levels) 與高空資料 (Pressure Levels)
    被正確下載並合併為單一檔案，以供模型使用。
    """
    def __init__(self, output_dir: str = "ncdb"):
        self.output_dir = output_dir
        self.retriever = CDSRetriever()
        self.processor = GribProcessor()
        self.validator = DataValidator()
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    def prepare_model_input(self, 
                          datetime_str: str, 
                          sl_request: Dict, 
                          pl_request: Dict, 
                          filename_prefix: str = "era5_input") -> str:
        """
        執行完整流程：下載 SL -> 下載 PL -> 合併 -> 清理 -> 驗證。
        
        Args:
            datetime_str: 用於檔名的時間標記 (e.g., '2023010100')
            sl_request: Surface Level 的 CDS 請求字典
            pl_request: Pressure Level 的 CDS 請求字典
        
        Returns:
            str: 最終合併後的 NetCDF 檔案路徑
        """
        temp_sl = os.path.join(self.output_dir, f"temp_sl_{datetime_str}.grib")
        temp_pl = os.path.join(self.output_dir, f"temp_pl_{datetime_str}.grib")
        final_nc = os.path.join(self.output_dir, f"{filename_prefix}_{datetime_str}.nc")

        downloaded_files = []

        try:
            # 1. 下載 Surface Level Data
            logger.info("Step 1/3: Downloading Surface Level Data...")
            self.retriever.retrieve('reanalysis-era5-single-levels', sl_request, temp_sl)
            downloaded_files.append(temp_sl)

            # 2. 下載 Pressure Level Data
            logger.info("Step 2/3: Downloading Pressure Level Data...")
            self.retriever.retrieve('reanalysis-era5-pressure-levels', pl_request, temp_pl)
            downloaded_files.append(temp_pl)

            # 3. 合併檔案 (GRIBs -> NC)
            logger.info("Step 3/3: Merging and Converting to NetCDF...")
            self.processor.merge_and_convert([temp_sl, temp_pl], final_nc)

            # 4. 驗證產出
            # 這裡可以定義模型需要的最小變數集，例如 SFNO 需要的變數
            # required_check = ['u', 'v', 't', 'z', 'r', 'u10', 'v10', 't2m', 'sp', 'msl', 'tcwv']
            self.validator.validate_nc(final_nc)
            
            logger.info(f"Dataset preparation complete: {final_nc}")
            return final_nc

        except DataPipelineError as e:
            logger.error(f"Pipeline failed: {e}")
            # 失敗時若產出了部分檔案，可選擇保留以便除錯，或在此清理
            raise e
        finally:
            # 5. 清理暫存的 GRIB 檔 (無論成功失敗都執行)
            self.processor.cleanup(downloaded_files)

if __name__ == "__main__":
    # Example usage for testing purposes
    # This block won't be executed when imported as a module
    import sys
    import pandas as pd # For dummy data generation only

    # 設定參數
    target_date = "2023-01-01"
    target_time = "00:00"
    output_directory = "ncdb_downloads_test" # Use a test directory
    
    # Surface Request Template
    sl_params = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure', 'surface_pressure'],
        'year': '2023', 'month': '01', 'day': '01', 'time': target_time,
        'area': [90, -180, -90, 180], # Global
    }

    # Pressure Request Template
    pl_params = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': ['u_component_of_wind', 'v_component_of_wind', 'geopotential', 'temperature', 'specific_humidity'],
        'pressure_level': ['1000', '850', '500', '200', '50'],
        'year': '2023', 'month': '01', 'day': '01', 'time': target_time,
        'area': [90, -180, -90, 180], # Global
    }

    manager = ERA5DataManager(output_dir=output_directory)

    try:
        # 執行完整流程
        # 這會生成一個包含 SL 和 PL 變數的單一 NetCDF 檔
        # 該檔案即可直接被 ModelLoader 讀取，用於單向降尺度或雙向耦合的初始場
        nc_file = manager.prepare_model_input(
            datetime_str="2023010100", 
            sl_request=sl_params, 
            pl_request=pl_params
        )
        print(f"SUCCESS: Ready for inference -> {nc_file}")

    except DataPipelineError as e:
        logger.error(f"FATAL: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Clean up the test directory
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
            print(f"Cleaned up test directory: {output_directory}")
