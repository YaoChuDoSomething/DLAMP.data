import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from src.core.pipeline import Task
from src.core.context import Context
# import dask
# from dask.diagnostics import ProgressBar


class Regridder(Task):
    def execute(self, context: Context):
        print("[INFO] Starting Regridder...")
        self.cfg = context.config

        # Configure Grid
        self.regrid_cfg = self.cfg["regrid"]
        self.method = self.regrid_cfg.get("method", "bilinear")
        use_dask = self.regrid_cfg.get("use_dask", True)
        n_workers = self.regrid_cfg.get("dask_workers", 4)
        _ = (use_dask, n_workers)  # Acknowledge to satisfy linter

        # Load Target Grid
        self.target_nc = self.regrid_cfg["target_nc"]
        with xr.open_dataset(self.target_nc, engine="netcdf4") as tgtds:
            self.XLONG = tgtds[self.regrid_cfg["target_lon"]].values
            self.XLAT = tgtds[self.regrid_cfg["target_lat"]].values
            self.static = tgtds[self.regrid_cfg["adopted_varlist"]]
            self.levels = self.regrid_cfg["levels"]  # Crucial for 3D outputs

        # 🔍 計算裁切邊界（目標區域 + buffer）
        buffer = self.regrid_cfg.get("crop_buffer_degrees", 2.0)
        self.lon_min = float(np.min(self.XLONG)) - buffer
        self.lon_max = float(np.max(self.XLONG)) + buffer
        self.lat_min = float(np.min(self.XLAT)) - buffer
        self.lat_max = float(np.max(self.XLAT)) + buffer
        print(
            f"[CROP] Target region: lon=[{self.lon_min:.2f}, {self.lon_max:.2f}], "
            f"lat=[{self.lat_min:.2f}, {self.lat_max:.2f}]"
        )

        # Timeline logic
        time_cfg = self.cfg["share"]["time_control"]
        start_t = datetime.strptime(time_cfg["start"], time_cfg["format"])
        end_t = datetime.strptime(time_cfg["end"], time_cfg["format"])
        base_step = timedelta(hours=time_cfg["base_step_hours"])  # 下載間隔（6hr）
        out_step = timedelta(
            hours=self.regrid_cfg["output_step_hours"]
        )  # 輸出間隔（1hr）

        # 判斷是否需要時間內插
        enable_temporal_interp = self.regrid_cfg.get("enable_temporal_interp", False)
        need_interp = (out_step < base_step) and enable_temporal_interp

        if need_interp:
            print(
                f"[INTERP] Temporal interpolation enabled: {base_step.total_seconds() / 3600}hr → {out_step.total_seconds() / 3600}hr"
            )

        total_steps = int((end_t - start_t) / out_step) + 1

        # I/O paths
        netcdf_dir = self.cfg["share"]["io_control"]["netcdf_dir"]
        prefix = self.cfg["share"]["io_control"]["prefix"]
        timestr_fmt = prefix["timestr_fmt"]

        if need_interp:
            print(
                f"[INTERP] Two-phase processing: {base_step.total_seconds() / 3600}hr → {out_step.total_seconds() / 3600}hr"
            )

            # === 階段1：處理所有邊界點（base_step 間隔） ===
            print("[PHASE 1] Processing boundary timesteps (base_step intervals)...")
            boundary_times = []
            curr = start_t
            while curr <= end_t:
                boundary_times.append(curr)
                timestamp = curr.strftime(timestr_fmt)
                input_nc = os.path.join(
                    netcdf_dir, f"{prefix['output']}_{timestamp}.nc"
                )
                regrid_nc = os.path.join(
                    netcdf_dir, f"{prefix['regrid']}_{timestamp}.nc"
                )

                if os.path.exists(input_nc) and not os.path.exists(regrid_nc):
                    self._process_single_timestep(input_nc, regrid_nc, curr)
                curr += base_step

            # === 階段2：批次時間內插 ===
            print("[PHASE 2] Batch temporal interpolation...")
            self.temporal_interp_batch(
                start_t, end_t, base_step, out_step, netcdf_dir, prefix, timestr_fmt
            )
        else:
            # 無時間內插：直接逐一處理
            total_steps = int((end_t - start_t) / out_step) + 1
            for i in range(total_steps):
                curr_time = start_t + out_step * i
                timestamp = curr_time.strftime(timestr_fmt)
                input_nc = os.path.join(
                    netcdf_dir, f"{prefix['output']}_{timestamp}.nc"
                )
                regrid_nc = os.path.join(
                    netcdf_dir, f"{prefix['regrid']}_{timestamp}.nc"
                )

                if os.path.exists(input_nc) and not os.path.exists(regrid_nc):
                    self._process_single_timestep(input_nc, regrid_nc, curr_time)

    def _process_single_timestep(self, input_nc, regrid_nc, curr_time):
        """處理單一時間點：水平內插 → 垂直內插"""
        print(f"[REGRID] Processing {curr_time}")
        with xr.open_dataset(input_nc) as src_ds:
            # 階段1：水平內插
            print(f"  [1/2] Horizontal regrid using {self.method}")
            ds_h = self.perform_regrid(src_ds, curr_time)

            # 階段2：垂直內插 (-lnP)
            print("  [2/2] Vertical interp in -ln(P) coordinate")
            ds_hv = self.vertical_interp_lnp(ds_h)

        ds_hv.to_netcdf(regrid_nc, format="NETCDF4")
        print(f"[DONE] Saved: {regrid_nc}")
        return regrid_nc

    def temporal_interp_batch(
        self, start_t, end_t, base_step, out_step, netcdf_dir, prefix, timestr_fmt
    ):
        """
        批次時間內插：處理整條時間段，3D 資料攤平成向量

        Args:
            start_t: 起始時間
            end_t: 結束時間
            base_step: 下載間隔（6hr）
            out_step: 輸出間隔（1hr）
            netcdf_dir: NetCDF 目錄
            prefix: 檔名前綴
            timestr_fmt: 時間字串格式
        """
        from scipy.interpolate import interp1d

        print("[TEMPORAL] Starting batch temporal interpolation...")

        # 1. 收集所有邊界時間點的檔案（已完成水平+垂直內插）
        boundary_times = []
        boundary_files = []
        curr = start_t
        while curr <= end_t:
            boundary_times.append(curr)
            ts = curr.strftime(timestr_fmt)
            regrid_nc = f"{netcdf_dir}/{prefix['regrid']}_{ts}.nc"
            if not os.path.exists(regrid_nc):
                print(f"[WARN] Missing boundary file: {regrid_nc}")
                return
            boundary_files.append(regrid_nc)
            curr += base_step

        print(f"[TEMPORAL] Loading {len(boundary_files)} boundary files...")

        # 2. 讀取所有邊界檔案
        datasets = [xr.open_dataset(f) for f in boundary_files]

        # 3. 建立時間軸（秒為單位）
        time_axis = np.array([(t - start_t).total_seconds() for t in boundary_times])

        # 4. 生成輸出時間序列
        output_times = []
        curr = start_t
        while curr <= end_t:
            output_times.append(curr)
            curr += out_step
        output_time_axis = np.array(
            [(t - start_t).total_seconds() for t in output_times]
        )

        # 5. 取得第一個 dataset 的結構
        ds0 = datasets[0]
        static_vars = ["XLONG", "XLAT", "HGT", "LANDMASK", "pres_levels"]

        # 6. 逐變數進行時間內插
        print(
            f"[TEMPORAL] Interpolating {len(ds0.data_vars)} variables to {len(output_times)} timesteps..."
        )

        # 準備輸出容器（每個時間點一個字典）
        output_data = {t: {} for t in output_times}

        # 靜態變數已經在 ds0 中，不需要額外處理

        # 處理動態變數
        for var in ds0.data_vars:
            if var in static_vars:
                continue  # 已處理

            print(f"  > Interpolating {var}")

            # 收集所有時間點的資料
            var_data = np.stack(
                [ds[var].isel(time=0).values for ds in datasets], axis=0
            )  # (n_boundary, ...)

            # 攤平空間維度
            if var_data.ndim == 4:  # (n_boundary, level, y, x)
                n_b, nl, ny, nx = var_data.shape
                var_flat = var_data.reshape(n_b, nl * ny * nx)
            elif var_data.ndim == 3:  # (n_boundary, y, x)
                n_b, ny, nx = var_data.shape
                var_flat = var_data.reshape(n_b, ny * nx)
            else:
                # 1D 或 2D 變數，直接跳過
                for t in output_times:
                    output_data[t][var] = ds0[var].values
                continue

            # Vectorized interpolation along axis 0 (time dimension of var_flat)
            # var_flat shape: (n_boundary, n_points)
            f = interp1d(
                time_axis,
                var_flat,
                kind="linear",
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            # f(output_time_axis) returns shape (n_out, n_points)
            var_interp_flat = f(output_time_axis).astype(np.float32)
            n_out = len(output_times)

            # 還原形狀並分配到各時間點
            if var_data.ndim == 4:
                var_interp = var_interp_flat.reshape(n_out, nl, ny, nx)
            else:
                var_interp = var_interp_flat.reshape(n_out, ny, nx)

            for t_idx, t in enumerate(output_times):
                # 取單一時間點，保持原始維度 (1, ...) 以匹配 ds0 的結構
                if var_data.ndim == 4:
                    output_data[t][var] = np.expand_dims(var_interp[t_idx], axis=0)
                else:
                    output_data[t][var] = np.expand_dims(var_interp[t_idx], axis=0)

        # 7. 儲存各時間點
        print("[TEMPORAL] Saving interpolated files...")
        for t in output_times:
            if t in boundary_times:
                # print(f"[TEMPORAL] Skipping boundary time {t} (file already exists)")
                continue

            ts = t.strftime(timestr_fmt)
            out_nc = f"{netcdf_dir}/{prefix['regrid']}_{ts}.nc"

            # 建立該時間點的 dataset（複製 ds0 的結構）
            out_ds = ds0.copy(deep=True)

            # 更新時間座標（直接修改 values）
            out_ds.coords["time"].values[:] = pd.Timestamp(t)

            # 更新變數資料
            for var in output_data[t]:
                if var in out_ds.data_vars:
                    out_ds[var].values = output_data[t][var]

            # 儲存
            out_ds.to_netcdf(out_nc, format="NETCDF4")
            print(f"[DONE] Saved: {out_nc}")

        # 8. 關閉所有檔案
        for ds in datasets:
            ds.close()

        print("[TEMPORAL] Batch temporal interpolation completed")

    def perform_regrid(self, src_ds, curr_time):
        if self.method == "bilinear":
            return self.regrid_bilinear(src_ds, curr_time)
        elif self.method == "idw":
            return self.regrid_idw(src_ds, curr_time)
        elif self.method == "kriging":
            return self.regrid_kriging(src_ds, curr_time)
        else:
            print(f"[WARN] Unknown method {self.method}, defaulting to bilinear")
            return self.regrid_bilinear(src_ds, curr_time)

    def vertical_interp_lnp(self, ds_h: xr.Dataset) -> xr.Dataset:
        """
        垂直內插：使用 -ln(P) 坐標系

        Args:
            ds_h: 已完成水平內插的資料集（含來源氣壓層）

        Returns:
            已完成垂直內插的資料集（目標氣壓層）
        """
        from scipy.interpolate import interp1d

        # 來源氣壓層（ERA5 下載的層）
        src_pres = self.regrid_cfg["levels"]
        # 目標氣壓層（從 target.nc 讀取）
        with xr.open_dataset(self.target_nc, engine="netcdf4") as tgtds:
            if self.regrid_cfg["target_pres"] in tgtds.variables:
                tgt_pres = tgtds[self.regrid_cfg["target_pres"]].values.tolist()
            else:
                tgt_pres = src_pres  # 若無目標氣壓層，使用來源層

        # 檢查是否需要垂直內插
        if np.array_equal(src_pres, tgt_pres):
            print(
                "[VERTICAL] Source and target pressure levels are identical, skipping vertical interpolation"
            )
            return ds_h

        # 轉換為 -ln(P)
        src_lnp = -np.log(np.array(src_pres))
        tgt_lnp = -np.log(np.array(tgt_pres))

        out_vars = {}

        for var in ds_h.data_vars:
            data = ds_h[var].values

            if data.ndim == 4:  # (time, level, y, x)
                nt, nl, ny, nx = data.shape
                data_v = np.empty((nt, len(tgt_pres), ny, nx), dtype=np.float32)

                # Vectorized interpolation along axis 1 (level dimension)
                # data shape: (nt, nl, ny, nx)
                f = interp1d(
                    src_lnp,
                    data,
                    kind="linear",
                    axis=1,
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                # Result shape will be (nt, n_out_levels, ny, nx)
                data_v = f(tgt_lnp).astype(np.float32)

                out_vars[var] = (
                    ["time", "pres_bottom_top", "south_north", "west_east"],
                    data_v,
                )

            elif data.ndim == 3:  # (time, y, x) - 地面變數
                out_vars[var] = (ds_h[var].dims, data)

        # Update pres_levels variable to match target pressure levels
        out_vars["pres_levels"] = (["pres_bottom_top"], tgt_pres)

        # 更新氣壓層座標
        coords = {
            "time": ds_h.coords["time"],
            "pres_bottom_top": (["pres_bottom_top"], tgt_pres),
            "south_north": ds_h.coords["south_north"],
            "west_east": ds_h.coords["west_east"],
        }

        # Logging dimensions

        return xr.Dataset(data_vars=out_vars, coords=coords)

    def regrid_bilinear(self, src_ds, curr_time):
        out_dict = {}

        # Config params
        src_lon_name = self.regrid_cfg["source_lon"]
        src_lat_name = self.regrid_cfg["source_lat"]

        # 🔍 裁切至目標區域（關鍵優化！）
        try:
            # 嘗試用 xarray 原生裁切（適用 1D 座標）
            src_cropped = src_ds.sel(
                {
                    src_lon_name: slice(self.lon_min, self.lon_max),
                    src_lat_name: slice(self.lat_min, self.lat_max),
                }
            )
            print(f"[CROP] Reduced grid from {src_ds.dims} to {src_cropped.dims}")
        except (KeyError, ValueError):
            # 若座標是 2D 或索引失敗，用條件篩選
            src_lon_vals = src_ds[src_lon_name].values
            src_lat_vals = src_ds[src_lat_name].values
            if src_lon_vals.ndim == 1:
                lon_mask = (src_lon_vals >= self.lon_min) & (
                    src_lon_vals <= self.lon_max
                )
                lat_mask = (src_lat_vals >= self.lat_min) & (
                    src_lat_vals <= self.lat_max
                )
                src_cropped = src_ds.isel(
                    {src_lon_name: lon_mask, src_lat_name: lat_mask}
                )
            else:
                print(
                    "[WARN] 2D coords detected, skipping crop (consider manual indexing)"
                )
                src_cropped = src_ds

        # Prepare source coordinates for griddata
        src_lon = src_cropped[src_lon_name].values
        src_lat = src_cropped[src_lat_name].values

        # Handle 1D vs 2D source coords
        if np.ndim(src_lon) == 1 and np.ndim(src_lat) == 1:
            lons, lats = np.meshgrid(src_lon, src_lat)
        else:
            lons, lats = src_lon, src_lat

        # ⚡ 用 column_stack 取代 zip（快 3-5x）
        points = np.column_stack([lons.ravel(), lats.ravel()])
        xi = (self.XLONG, self.XLAT)

        nt, ny, nx = np.shape(
            self.XLONG
        )  # actually 1, ny, nx usually if from target_nc
        # Check logic: if XLONG is (1, ny, nx), then shape is ok.
        # usually target_nc vars are (time, y, x)

        # Dimensions for Output
        dim_upp = ["time", "pres_bottom_top", "south_north", "west_east"]
        dim_sfc = ["time", "south_north", "west_east"]

        # Interpolate each variable in source dataset
        for var in src_cropped.data_vars:
            data = np.squeeze(src_cropped[var].values)
            # print(f"  > Interpolating {var}, shape: {data.shape}")

            if data.ndim == 3:  # (Level, Lat, Lon) assuming time squeezed out or 1
                nl = data.shape[0]
                data_h = np.empty((nl, ny, nx))  # new grid shape
                for pl in range(nl):
                    # Linear interpolation (scipy griddata)
                    data_h[pl] = griddata(points, data[pl].ravel(), xi, method="linear")
                # Expand dims to match (time, Level, Y, X)
                data_h = np.expand_dims(data_h, axis=0)
                out_dict[var] = (dim_upp, data_h.astype(np.float32))

            elif data.ndim == 2:  # (Lat, Lon)
                data_h = griddata(points, data.ravel(), xi, method="linear")
                # Make sure it matches (ny, nx) and add time dim
                # griddata returns shape of xi. if xi is (ny, nx), result is (ny, nx)
                # But xi passed as (XLONG, XLAT) which are (1, ny, nx) or (ny, nx)?
                # Legacy code: xi = (self.XLONG, self.XLAT)
                # target_nc loading: self.XLONG = tgtds[self.tgtlon].values
                # if target NC has time, it might be (1, ny, nx).
                # Let's trust legacy code structure:
                # data_h = np.expand_dims(np.reshape(data_h, (ny,nx)), axis=0)

                # Careful with shape matching.
                # If XLONG is (1, ny, nx), griddata result might be (1, ny, nx).
                # We want to ensure it ends up as (1, ny, nx).

                data_h = data_h.reshape(ny, nx)  # forcing safe reshape
                data_h = np.expand_dims(data_h, axis=0)
                out_dict[var] = (dim_sfc, data_h.astype(np.float32))

        # Add Static Variables from Target Grid
        for var in self.static.data_vars:
            static_data = self.static[var].values
            if static_data.ndim == 2:
                out_dict[var] = (dim_sfc, np.expand_dims(static_data, axis=0))
            else:
                # If it already has time dimension or 3D
                # Just passing it through as-is (legacy behavior)
                out_dict[var] = (self.static[var].dims, static_data)

        # Coordinate Variables
        out_dict["XLONG"] = (dim_sfc, np.expand_dims(np.squeeze(self.XLONG), axis=0))
        out_dict["XLAT"] = (dim_sfc, np.expand_dims(np.squeeze(self.XLAT), axis=0))
        out_dict["pres_levels"] = (["pres_bottom_top"], self.levels)

        # Construct Dataset
        outds = xr.Dataset(
            data_vars=out_dict,
            coords={
                "time": ("time", [pd.Timestamp(curr_time)]),
                "pres_bottom_top": ("pres_bottom_top", self.levels),
                "south_north": ("south_north", np.arange(ny)),
                "west_east": ("west_east", np.arange(nx)),
            },
            attrs={"title": f"Regridded ERA5 at {curr_time}"},
        )
        return outds

    def regrid_idw(self, src_ds, curr_time):
        print("[INFO] IDW Reserved")
        return xr.Dataset()

    def regrid_kriging(self, src_ds, curr_time):
        print("[INFO] Kriging Reserved")
        return xr.Dataset()
