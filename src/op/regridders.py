import xarray as xr
import numpy as np
from typing import Dict, List, Tuple
from .utils.file_utils import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""

    pass


# ------------------------------------------------------------------------------
# 1. Data Standardizer (變數標準化與檢查)
# ------------------------------------------------------------------------------


class DataStandardizer:
    """
    負責將輸入資料的變數名稱、單位與維度標準化。
    確保資料進入數學運算前已符合定義好的 Schema。
    """

    def __init__(self, var_mapping: Dict[str, str], required_dims: List[str] = None):
        """
        Args:
            var_mapping: 字典，格式為 {'原始名稱': '標準名稱'}
            required_dims: 列表，例如 ['time', 'level', 'lat', 'lon']
        """
        self.var_mapping = var_mapping
        self.required_dims = required_dims or ["time", "level", "lat", "lon"]

    def process(self, ds: xr.Dataset) -> xr.Dataset:
        logger.info("Standardizing dataset variables and dimensions...")
        ds_out = ds.copy()

        # 1. Rename Variables
        rename_dict = {k: v for k, v in self.var_mapping.items() if k in ds_out}
        if rename_dict:
            ds_out = ds_out.rename(rename_dict)
            logger.info(f"Renamed variables: {rename_dict}")

        # 2. Check Dimensions
        # 如果是 2D 區域網格，可能會有 x, y 而非 lat, lon，需視情況彈性處理
        # 這裡假設輸入的全球模式資料必須是標準 lat/lon
        missing_dims = [d for d in self.required_dims if d not in ds_out.dims]
        if missing_dims:
            # 有些地面資料可能沒有 level，這不是錯誤，視具體變數而定
            # 這裡僅做警告或針對特定變數檢查
            logger.debug(f"Note: Dataset missing standard dimensions: {missing_dims}")

        # 3. Ensure Standard Units (Example: Geopotential to Geopotential Height)
        # 這裡示範將位勢 (m^2/s^2) 轉為 位勢高度 (gpm)
        if "z" in ds_out:  # 假設 z 是位勢
            ds_out["z"] = ds_out["z"] / 9.80665
            ds_out["z"].attrs["units"] = "gpm"
            ds_out = ds_out.rename({"z": "gh"})  # Rename to Geopotential Height
            logger.info("Converted Geopotential (z) to Geopotential Height (gh).")

        return ds_out


# ------------------------------------------------------------------------------
# 2. Time Upsampler (時間插值)
# ------------------------------------------------------------------------------


class TimeUpsampler:
    """
    負責將時間解析度不足的資料 (如 6H) 內插至目標解析度 (如 1H)。
    """

    def __init__(self, target_freq: str = "1H"):
        self.target_freq = target_freq

    def process(self, ds: xr.Dataset) -> xr.Dataset:
        logger.info(f"Upsampling time dimension to {self.target_freq}...")

        if "time" not in ds.coords:
            raise PreprocessingError("Dataset missing 'time' coordinate.")

        # 檢查是否已經符合頻率，避免不必要的計算
        try:
            current_freq = xr.infer_freq(ds.time)
            if current_freq == self.target_freq:
                logger.info("Time frequency matches target. Skipping interpolation.")
                return ds
        except Exception:
            pass  # 無法推斷頻率時繼續執行

        try:
            # 使用 xarray 的 resample + interpolate
            # 這比手寫 scipy.interp1d 更能處理 Dask Array 且程式碼更簡潔
            # method='linear' 適合大氣變數的連續變化
            ds_upsampled = ds.resample(time=self.target_freq).interpolate("linear")

            # 確保不超出原始時間範圍 (Extrapolation 在氣象上通常危險，這裡傾向於不外推)
            # 若需填補頭尾，可使用 bfill/ffill

            logger.info(f"Time upsampling complete. New size: {ds_upsampled.time.size}")
            return ds_upsampled
        except Exception as e:
            raise PreprocessingError(f"Time interpolation failed: {e}")


# ------------------------------------------------------------------------------
# 3. Vertical Transformer (垂直座標轉換)
# ------------------------------------------------------------------------------


class VerticalTransformer:
    """
    負責處理垂直座標。
    計算對數氣壓座標 (Log-Pressure Coordinate) 並附加到 Dataset。
    Z* = -H * ln(p / p_ref)
    """

    def __init__(self, scale_height_km: float = 7.0, ref_pressure_hpa: float = 1000.0):
        self.H = scale_height_km * 1000.0  # Convert to meters
        self.p_ref = ref_pressure_hpa

    def process(self, ds: xr.Dataset) -> xr.Dataset:
        logger.info("Calculating vertical log-pressure coordinates...")

        # 尋找壓力座標名稱 (通常是 level, pressure_level, isobaricInhPa)
        p_dim = None
        for dim in ["level", "pressure_level", "isobaricInhPa"]:
            if dim in ds.coords:
                p_dim = dim
                break

        if p_dim is None:
            logger.warning(
                "No pressure dimension found. Skipping vertical transformation."
            )
            return ds

        # 取得壓力值 (確保單位一致，假設輸入 level 單位為 hPa，若為 Pa 需調整)
        p_values = ds[p_dim]

        # 簡單檢查單位：如果數值很大 (>2000)，可能是 Pa，否則假設為 hPa
        if p_values.max() > 2000:
            p_in_hpa = p_values / 100.0
        else:
            p_in_hpa = p_values

        # 計算 Log-Pressure Height
        # Z* = -H * ln(p / p_s)
        z_star = -self.H * np.log(p_in_hpa / self.p_ref)

        # 將計算結果作為座標加入 Dataset
        ds_out = ds.assign_coords(log_pressure_height=(p_dim, z_star.values))
        ds_out["log_pressure_height"].attrs = {
            "long_name": "Log-pressure height",
            "units": "m",
            "formula": f"-{self.H} * ln(p / {self.p_ref})",
        }

        logger.info(
            f"Added 'log_pressure_height' coordinate based on dimension '{p_dim}'."
        )
        return ds_out


# ------------------------------------------------------------------------------
# 4. Spatial Regridder (空間降尺度優化)
# ------------------------------------------------------------------------------


class SpatialRegridder:
    """
    負責將全球網格 (Lat/Lon) 轉換為 區域網格 (XY Equidistant)。
    優化策略：
    1. 計算區域網格的 Bounding Box (Lat/Lon 範圍)。
    2. 先裁切 (Slice) 全球資料，只保留相關區域 (加上 Buffer)。
    3. 再進行精細的空間內插。
    """

    def __init__(self, buffer_deg: float = 2.0):
        self.buffer = buffer_deg

    def _get_bounding_box(
        self, target_ds: xr.Dataset
    ) -> Tuple[float, float, float, float]:
        """計算目標網格的經緯度範圍。"""
        # 目標網格必須包含 2D 的 lat, lon 變數
        if "lat" not in target_ds.coords and "lat" not in target_ds.data_vars:
            raise PreprocessingError("Target grid missing 'lat' variable.")
        if "lon" not in target_ds.coords and "lon" not in target_ds.data_vars:
            raise PreprocessingError("Target grid missing 'lon' variable.")

        min_lat = float(target_ds["lat"].min())
        max_lat = float(target_ds["lat"].max())
        min_lon = float(target_ds["lon"].min())
        max_lon = float(target_ds["lon"].max())

        return min_lat, max_lat, min_lon, max_lon

    def process(self, source_ds: xr.Dataset, target_grid_ds: xr.Dataset) -> xr.Dataset:
        logger.info("Performing spatial regridding (Global -> Regional)...")

        # 1. 計算目標範圍並加上 Buffer
        min_lat, max_lat, min_lon, max_lon = self._get_bounding_box(target_grid_ds)

        slice_lat_min = min_lat - self.buffer
        slice_lat_max = max_lat + self.buffer
        slice_lon_min = min_lon - self.buffer
        slice_lon_max = max_lon + self.buffer

        logger.info(
            f"Target ROI: Lat[{min_lat:.2f}, {max_lat:.2f}], Lon[{min_lon:.2f}, {max_lon:.2f}]"
        )
        logger.info(
            f"Slicing source with buffer: Lat[{slice_lat_min:.2f}, {slice_lat_max:.2f}]"
        )

        # 2. 裁切原始資料 (Subsetting) - 這是效能關鍵
        # 注意：全球資料的 lat 排序可能是遞增或遞減，需處理
        src_lat_increasing = source_ds["lat"][1] > source_ds["lat"][0]

        if src_lat_increasing:
            lat_slice = slice(slice_lat_min, slice_lat_max)
        else:
            lat_slice = slice(slice_lat_max, slice_lat_min)

        # 處理經度 0/360 跨越問題 (這裡簡化處理，假設皆為 -180~180 或 0~360 對齊)
        # 若需嚴謹處理跨越子午線，需使用 roll 或更複雜的邏輯
        lon_slice = slice(slice_lon_min, slice_lon_max)

        try:
            ds_subset = source_ds.sel(lat=lat_slice, lon=lon_slice)

            if ds_subset.lat.size == 0 or ds_subset.lon.size == 0:
                raise PreprocessingError(
                    "Resulting subset is empty. Check coordinate ranges."
                )
        except Exception as e:
            raise PreprocessingError(f"Slicing failed: {e}")

        # 3. 空間內插 (Interpolation)
        # 目標網格是 XY 等距，但有對應的 Lat/Lon 2D 陣列
        # xarray 的 interp 支援傳入 2D 的 lat/lon 座標進行重採樣
        try:
            # 這裡假設 target_grid_ds 的 lat/lon 是 2D 變數 (y, x)
            # 我們需要將其作為座標傳入 interp

            # 確保目標座標是 DataArray 格式
            tgt_lat = target_grid_ds["lat"]
            tgt_lon = target_grid_ds["lon"]

            regridded_ds = ds_subset.interp(
                lat=tgt_lat,
                lon=tgt_lon,
                method="linear",  # 雙線性內插
                kwargs={"fill_value": "extrapolate"},  # 避免邊界 NaN
            )

            # 內插後的 dataset 會繼承 lat/lon 的維度 (y, x)
            # 刪除舊的 lat/lon 索引，保留新的 x, y 座標
            logger.info(
                f"Spatial interpolation complete. Output shape: {regridded_ds.dims}"
            )
            return regridded_ds

        except Exception as e:
            raise PreprocessingError(f"Spatial interpolation failed: {e}")


# ------------------------------------------------------------------------------
# Workflow Orchestrator (管線整合)
# ------------------------------------------------------------------------------


class DataPreparationPipeline:
    """
    將上述步驟串聯成完整工作流。
    """

    def __init__(self, config: Dict):
        self.standardizer = DataStandardizer(var_mapping=config.get("var_map", {}))
        self.time_upsampler = TimeUpsampler(
            target_freq=config.get("time_target_freq", "1H")
        )
        self.vertical_transformer = VerticalTransformer(
            scale_height_km=config.get("scale_height_km", 7.0),
            ref_pressure_hpa=config.get("ref_pressure_hpa", 1000.0),
        )
        self.spatial_regridder = SpatialRegridder(
            buffer_deg=config.get("roi_buffer", 2.0)
        )

    def run(self, source_ds: xr.Dataset, target_grid_ds: xr.Dataset) -> xr.Dataset:
        logger.info("Starting Data Preparation Pipeline...")

        # 1. Standardize
        ds = self.standardizer.process(source_ds)

        # 2. Time Upsample (6H -> 1H)
        ds = self.time_upsampler.process(ds)

        # 3. Vertical Coord (Add Log-Pressure)
        ds = self.vertical_transformer.process(ds)

        # 4. Spatial Regrid (Global -> Regional with Optimization)
        ds = self.spatial_regridder.process(ds, target_grid_ds)

        logger.info("Pipeline completed successfully.")
        return ds


if __name__ == "__main__":
    import pandas as pd  # For dummy data generation only
    import sys

    # 模擬資料與配置
    config = {
        "var_map": {
            "t": "temperature",
            "u": "u_wind",
            "v": "v_wind",
            "z": "geopotential",
        },
        "roi_buffer": 3.0,
        "time_target_freq": "1H",
    }

    # 1. 建立模擬的全球資料 (6-hourly, Pressure levels, Lat/Lon)
    times = pd.date_range("2023-01-01", periods=4, freq="6H")
    levels = [1000, 850, 500]
    lats = np.linspace(-10, 40, 50)  # 包含台灣區域
    lons = np.linspace(100, 140, 40)

    data = np.random.rand(len(times), len(levels), len(lats), len(lons))
    global_ds = xr.Dataset(
        data_vars={
            "t": (("time", "level", "lat", "lon"), data * 300),
            "z": (("time", "level", "lat", "lon"), data * 10000),
        },
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )

    # 2. 建立模擬的區域目標網格 (XY Equidistant, 2D Lat/Lon)
    # 假設這是一個以台灣為中心的 2km 解析度網格
    x = np.linspace(0, 100000, 50)  # meters
    y = np.linspace(0, 100000, 50)  # meters
    X, Y = np.meshgrid(x, y)

    # 簡單模擬 Lat/Lon 隨 X/Y 的變化
    target_lats = 22.0 + Y / 111000.0
    target_lons = 120.0 + X / (111000.0 * np.cos(np.deg2rad(22.0)))

    target_grid_ds = xr.Dataset(
        coords={
            "x": x,
            "y": y,
            "lat": (("y", "x"), target_lats),
            "lon": (("y", "x"), target_lons),
        }
    )

    # 3. 執行管線
    pipeline = DataPreparationPipeline(config)

    try:
        final_ds = pipeline.run(global_ds, target_grid_ds)
        print("\nFinal Processed Dataset:")
        print(final_ds)

        # 驗證
        assert (
            final_ds.time.size == 19
        )  # 4 steps 6H apart -> 18 hours span -> 19 hourly steps
        assert "log_pressure_height" in final_ds.coords
        assert final_ds.dims["x"] == 50
        assert final_ds.dims["y"] == 50

    except PreprocessingError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
