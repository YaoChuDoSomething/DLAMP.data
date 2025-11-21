import xarray as xr
import torch
import os
import pandas as pd
from earth2studio.models.px import SFNO
from .utils.file_utils import get_logger

logger = get_logger(__name__)


class GlobalOperatorsError(Exception):
    """Custom exception for global_operators errors."""

    pass


# ------------------------------------------------------------------------------
# 1. Data Loading & Standardization (資料讀取與標準化)
# ------------------------------------------------------------------------------


class VariableMapper:
    """
    負責處理變數名稱映射與維度標準化。
    獨立於檔案讀取之外，便於測試與重用。
    """

    def __init__(self, variable_map: dict = None):
        self.variable_map = variable_map or {}

    def standardize(self, ds: xr.Dataset) -> xr.Dataset:
        """將輸入 Dataset 的變數名稱與座標轉換為模型標準格式。"""
        processed_vars = {}

        for var_name, var_data in ds.data_vars.items():
            # 處理變數更名與 Level 擴展
            if var_name in self.variable_map:
                mapping_info = self.variable_map[var_name]
                new_name = (
                    mapping_info
                    if isinstance(mapping_info, str)
                    else mapping_info.get("new_name", var_name)
                )
                level = (
                    mapping_info.get("level")
                    if isinstance(mapping_info, dict)
                    else None
                )

                new_var_data = var_data.copy()
                new_var_data.name = new_name

                if level is not None and "level" not in new_var_data.dims:
                    new_var_data = new_var_data.expand_dims("level").assign_coords(
                        level=[level]
                    )

                processed_vars[new_name] = new_var_data
            else:
                processed_vars[var_name] = var_data

        # 重組 Dataset 並處理座標
        new_ds = xr.Dataset(processed_vars)
        # 繼承原有的座標 (如 time, lat, lon)
        for coord in ds.coords:
            if coord in new_ds.coords:
                continue
            # 簡單處理：若原座標對應的維度存在於新變數中，則保留
            if any(coord in v.dims for v in new_ds.data_vars.values()):
                new_ds = new_ds.assign_coords({coord: ds[coord]})

        return new_ds


class ModelLoader:
    """
    負責載入模型權重與原始檔案 I/O。
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.logger = get_logger(__name__)

    def load_sfno_model(self) -> torch.nn.Module:
        self.logger.info("ModelLoader: Loading SFNO model...")
        try:
            package = SFNO.load_default_package()
            model = SFNO.load_model(package).to(self.device)
            return model
        except Exception as e:
            raise GlobalOperatorsError(f"Error loading SFNO model: {e}")

    def load_nc_file(self, path: str) -> xr.Dataset:
        self.logger.info(f"ModelLoader: Reading NetCDF from {path}...")
        if not os.path.exists(path):
            raise GlobalOperatorsError(f"File not found: {path}")
        try:
            return xr.open_dataset(path)
        except Exception as e:
            raise GlobalOperatorsError(f"Error reading NetCDF: {e}")


# ------------------------------------------------------------------------------
# 2. Inference Logic (推論邏輯)
# ------------------------------------------------------------------------------


class InferenceEngine:
    """
    負責執行模型的數值計算。
    支援 Generator 模式，允許外部控制每一步驟 (便於耦合)。
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = model.device
        self.logger = get_logger(__name__)

    def _prepare_input(self, current_state: xr.Dataset) -> tuple[torch.Tensor, dict]:
        """內部方法：將 xarray 轉換為模型所需的 Tensor"""
        input_coords_info = self.model.input_coords()
        required_vars = input_coords_info["variable"]

        # 檢查變數完整性
        missing = [v for v in required_vars if v not in current_state.data_vars]
        if missing:
            raise GlobalOperatorsError(f"Missing variables for inference: {missing}")

        # 構建 Tensor (B, C, H, W)
        ordered_das = [current_state[var] for var in required_vars]
        input_da = xr.concat(ordered_das, dim="variable")

        x = torch.from_numpy(input_da.values).unsqueeze(0).float().to(self.device)

        time_val = pd.to_datetime(current_state["time"].values).isoformat()
        if isinstance(time_val, list):
            time_val = time_val[0]  # Handle single time vs list

        coords = {"time": [time_val], "variable": required_vars}
        return x, coords

    def create_iterator(self, initial_state: xr.Dataset):
        """
        建立推論迭代器。
        Yields:
            (xr.Dataset): 每一步的預報結果
        """
        x, coords = self._prepare_input(initial_state)

        # 使用 earth2studio 的迭代器
        iterator = self.model.create_iterator(x, coords)

        # 取得靜態座標資訊用於重建 xarray
        lat = initial_state["lat"]
        lon = initial_state["lon"]

        for i, (x_out, coords_out) in enumerate(iterator):
            # 略過第 0 步 (初始狀態)，或者根據需求決定是否回傳
            if i == 0:
                continue

            # 將 Tensor 轉回 xarray
            ds_out = xr.DataArray(
                data=x_out.squeeze(0).cpu().numpy(),
                dims=("variable", "lat", "lon"),
                coords={
                    "variable": coords_out["variable"],
                    "lat": lat,
                    "lon": lon,
                    "lead_time": coords_out["lead_time"],
                },
            ).to_dataset(dim="variable")

            # 計算絕對時間
            current_lead = pd.Timedelta(coords_out["lead_time"][0])
            start_time = pd.to_datetime(coords["time"][0])
            valid_time = start_time + current_lead

            ds_out = ds_out.assign_coords(time=valid_time)
            yield ds_out


# ------------------------------------------------------------------------------
# 3. Coupling & Feedback (耦合與回饋)
# ------------------------------------------------------------------------------


class FeedbackCoupler:
    """
    負責雙向回饋：接收高解析度區域資料，更新全球場。
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha (float): Nudging 係數 (0~1)，控制區域資料對全球場的影響程度。
        """
        self.alpha = alpha
        self.logger = get_logger(__name__)

    def blend_fields(
        self, global_ds: xr.Dataset, regional_ds: xr.Dataset
    ) -> xr.Dataset:
        """
        將區域模式的結果融合進全球模式狀態中。

        Args:
            global_ds: 當前全球模式預報 (Low Res)
            regional_ds: 當前區域模式分析/預報 (High Res)

        Returns:
            updated_global_ds: 更新後的全球場
        """
        self.logger.info(
            "FeedbackCoupler: Blending regional data into global fields..."
        )

        # 1. 將區域資料降尺度/插值到全球網格 (Upscaling)
        # 注意：實際應用中需處理邊界平滑 (Tapering) 以避免數值震盪
        try:
            regional_on_global_grid = regional_ds.interp_like(
                global_ds, method="linear"
            )

            # 2. 找出重疊區域的 Mask (非 NaN 的地方)
            # 假設 regional_ds 在非區域範圍是 NaN
            mask = regional_on_global_grid.notnull()

            # 3. 混合公式: Global = Global * (1-a) + Regional * a
            # 只在 Mask 為 True 的地方更新
            updated_ds = global_ds.copy()

            for var in global_ds.data_vars:
                if var in regional_on_global_grid:
                    diff = regional_on_global_grid[var] - global_ds[var]
                    # 簡單的 Nudging
                    updated_ds[var] = global_ds[var] + (diff * self.alpha).where(
                        mask[var], 0
                    )

            return updated_ds

        except Exception as e:
            self.logger.warning(
                f"Blending failed: {e}. Returning original global state."
            )
            return global_ds


# ------------------------------------------------------------------------------
# 4. Output & Downscaling (輸出與降尺度)
# ------------------------------------------------------------------------------


class DataExporter:
    """
    負責資料輸出，包含垂直積分與區域裁切 (單向降尺度準備)。
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = get_logger(__name__)

    def extract_region(self, ds: xr.Dataset, roi: dict) -> xr.Dataset:
        """
        單向降尺度：裁切特定區域以供區域模式使用。
        Args:
            roi (dict): {'lat_min': -10, 'lat_max': 10, 'lon_min': 100, 'lon_max': 120}
        """
        if not roi:
            return ds

        self.logger.info(f"DataExporter: Extracting region {roi}...")
        try:
            # 處理經度 0-360 或 -180-180 的問題 (略，假設一致)
            region_ds = ds.sel(
                lat=slice(roi["lat_max"], roi["lat_min"]),  # SFNO 通常 lat 是從北到南
                lon=slice(roi["lon_min"], roi["lon_max"]),
            )
            return region_ds
        except Exception as e:
            self.logger.error(f"Region extraction failed: {e}")
            raise GlobalOperatorsError(e)

    def save(self, ds: xr.Dataset, prefix: str):
        """儲存檔案"""
        if "time" in ds.coords:
            # 取第一個時間點作為檔名
            time_str = pd.to_datetime(ds.time.values).strftime("%Y%m%d%H%M")
        else:
            time_str = "static"

        filename = f"{prefix}_{time_str}.nc"
        path = os.path.join(self.output_dir, filename)

        self.logger.info(f"DataExporter: Saving to {path}")
        ds.to_netcdf(path)


# No __main__ block here, as it's a module
