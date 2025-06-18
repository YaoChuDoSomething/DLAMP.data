import numpy as np
import xarray as xr

# 輔助函數：用於創建 DataArray，減少重複程式碼
def _create_dataarray(data: np.ndarray, ds: xr.Dataset, var_name: str, long_name: str, units: str) -> xr.DataArray:
    """
    建立一個 xarray.DataArray，並根據資料維度從輸入 Dataset 推斷座標。
    """
    
    # 根據資料的維度來推斷 DataArray 的 dims 和 coords
    # 這裡假設診斷變數的輸出維度與其輸入變數的維度結構相似
    # 並且會根據 data.ndim 來選擇合適的座標
    
    data_dims = []
    data_coords = {}

    # 4D 變數通常是 (Time, pres_bottom_top, south_north, west_east)
    if data.ndim == 4:
        data_dims = ["Time", "pres_bottom_top", "south_north", "west_east"]
        data_coords["Time"] = ds["Time"]
        data_coords["pres_bottom_top"] = ds["pres_bottom_top"]
        data_coords["south_north"] = ds["south_north"]
        data_coords["west_east"] = ds["west_east"]
    # 3D 變數可能是 (Time, south_north, west_east) (地表變數的時間序列)
    # 或者 (pres_bottom_top, south_north, west_east) (單一時間步的垂直變數)
    elif data.ndim == 3:
        # 這裡需要更精確的判斷。如果診斷變數有壓力層次，但輸出壓縮了時間維度，
        # 則其形狀的第一個維度應與 ds["pres_bottom_top"] 的長度一致。
        # 如果診斷變數是地表變數，則其形狀應為 (時間, 緯度, 經度)。
        
        # 為了簡化，我們假設如果診斷函數返回 3D 陣列，它就是 (Time, south_north, west_east)
        # 如果需要處理 (pres_bottom_top, south_north, west_east) 且沒有時間維度的情況，
        # 診斷函數應該返回一個正確維度的 DataArray，或者在外部進行 expand_dims。
        
        # 目前的邏輯是，如果診斷結果是 3D，且 ds 中有 'Time' 變數，就認為它是 Time 維度，
        # 否則，如果 ds 中有 'pres_bottom_top' 且其長度與資料的第一維度匹配，則認為是壓力維度。
        
        # 更通用的方式是，如果診斷變數的輸入是 4D，輸出 4D；如果輸入是 3D，輸出 3D。
        # 在這裡，我們假設對於 3D 輸出，它通常是地表變數（時間、緯度、經度）。
        # 如果有垂直層次的 3D 輸出，建議在診斷函數內部明確處理其維度。

        # 這裡根據常見情境做一個判斷：
        if "Time" in ds.dims and ds["Time"].shape[0] == data.shape[0]:
            data_dims = ["Time", "south_north", "west_east"]
            data_coords["Time"] = ds["Time"]
            data_coords["south_north"] = ds["south_north"]
            data_coords["west_east"] = ds["west_east"]
        elif "pres_bottom_top" in ds.dims and ds["pres_bottom_top"].shape[0] == data.shape[0]:
            data_dims = ["pres_bottom_top", "south_north", "west_east"]
            data_coords["pres_bottom_top"] = ds["pres_bottom_top"]
            data_coords["south_north"] = ds["south_north"]
            data_coords["west_east"] = ds["west_east"]
        else:
            # 如果以上判斷都不符合，可能是其他 3D 結構，需要更具體的邏輯
            # 或者資料維度錯誤
            raise ValueError(f"Could not infer dimensions for 3D data: {var_name} with shape {data.shape}")
    # 2D 變數通常是 (south_north, west_east) (靜態變數或單一時間步的地表變數)
    elif data.ndim == 2:
        data_dims = ["south_north", "west_east"]
        data_coords["south_north"] = ds["south_north"]
        data_coords["west_east"] = ds["west_east"]
        
        # 如果是靜態變數，它可能沒有 Time 維度
        # 但在 DataRegridder 中，我們期望所有的診斷輸出都是有 Time 維度的
        # 所以對於 2D 靜態資料，需要擴展一個 Time 維度
        # 注意：這裡應該由 DataRegridder 的 `process_single_time` 處理靜態變數的 Time 維度
        # 診斷函數通常處理包含 Time 維度的資料。
        # 如果診斷函數的輸出是 2D (例如單一時間點的結果)，但最終需要 3D (Time, Y, X)，
        # 則在創建 DataArray 時，應明確添加時間維度。
        # 為了統一輸出 DataArray 的結構，這裡將 2D 擴展為 3D，假設 Time 維度是第一個
        
        # 如果資料是單一時間步的地表資料，並且 ds 有 Time 維度
        if "Time" in ds.dims:
            data = np.expand_dims(data, axis=0) # 變成 (1, south_north, west_east)
            data_dims.insert(0, "Time") # 在維度列表中前面插入 "Time"
            data_coords["Time"] = ds["Time"] # 加入 Time 座標

    # 建立 DataArray
    da = xr.DataArray(
        data,
        coords=data_coords,
        dims=data_dims,
        name=var_name,
        attrs={
            "long_name": long_name,
            "units": units,
        }
    )
    return da

def diag_HGT(ds: xr.Dataset) -> xr.DataArray:
    """
    [Static Variable] Terrain Height 
    """
    data = np.squeeze(ds["HGT"].values)
    return _create_dataarray(data, ds, "HGT", "Terrain Height", "m")

def diag_LANDMASK(ds: xr.Dataset) -> xr.DataArray:
    """
    [Static Variable] Land-Sea Mask
    """
    data = np.squeeze(ds["LANDMASK"].values)
    return _create_dataarray(data, ds, "LANDMASK", "land-sea mask (land=1, sea=0)", "1")

def diag_z_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Geopotential height = geopotential / g
    """
    data = np.squeeze(ds["z"].values / 9.81)
    return _create_dataarray(data, ds, "z_p", "Geopotential Height", "m")

def diag_tk_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Air Temperature [K]
    """
    data = np.squeeze(ds["t"].values)
    return _create_dataarray(data, ds, "tk_p", "Air Temperature", "K")

def diag_umet_p(ds: xr.Dataset) -> xr.DataArray:
    """
    U-component of wind
    """
    data = np.squeeze(ds["u"].values)
    return _create_dataarray(data, ds, "umet_p", "U-component of Wind", "m/s")

def diag_vmet_p(ds: xr.Dataset) -> xr.DataArray:
    """
    V-component of wind
    """
    data = np.squeeze(ds["v"].values)
    return _create_dataarray(data, ds, "vmet_p", "V-component of Wind", "m/s")

def diag_QVAPOR_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific humidity to water vapor mixing ratio
    """
    data = np.squeeze(ds["q"].values) / (1 - np.squeeze(ds["q"].values))
    return _create_dataarray(data, ds, "QVAPOR_p", "Water Vapor Mixing Ratio", "kg/kg")

def diag_QRAIN_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific rain water content to rain water mixing ratio
    """
    data = np.squeeze(ds["crwc"].values) / (1 - np.squeeze(ds["crwc"].values))
    return _create_dataarray(data, ds, "QRAIN_p", "Rain Water Mixing Ratio", "kg/kg")

def diag_QCLOUD_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific cloud liquid water content to cloud water mixing ratio
    """
    data = np.squeeze(ds["clwc"].values) / (1 - np.squeeze(ds["clwc"].values))
    return _create_dataarray(data, ds, "QCLOUD_p", "Cloud Water Mixing Ratio", "kg/kg")

def diag_QSNOW_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific snow water content to snow water mixing ratio
    """
    data = np.squeeze(ds["cswc"].values) / (1 - np.squeeze(ds["cswc"].values))
    return _create_dataarray(data, ds, "QSNOW_p", "Snow Water Mixing Ratio", "kg/kg")

def diag_QICE_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific cloud ice content to cloud ice mixing ratio
    """
    data = np.squeeze(ds["ciwc"].values) / (1 - np.squeeze(ds["ciwc"].values))
    return _create_dataarray(data, ds, "QICE_p", "Cloud Ice Mixing Ratio", "kg/kg")

def diag_QGRAUP_p(ds: xr.Dataset) -> xr.DataArray:
    """
    Specific graupel water content to graupel water mixing ratio
    """
    # 這裡的診斷邏輯是 ds["q"].values * 0，如果其維度應該與 ds["q"] 相同，
    # 則應該使用 ds["q"] 來推斷維度。
    data = np.squeeze(ds["q"].values * 0)
    return _create_dataarray(data, ds, "QGRAUP_p", "Graupel Water Mixing Ratio", "kg/kg")

def diag_wa_p(ds: xr.Dataset) -> xr.DataArray:
    """
    omega [Pa s-1] to w [m s-1]
    """
    tmk = np.squeeze(ds["t"].values)
    qvp = np.squeeze(ds["q"].values) / (1 - np.squeeze(ds["q"].values))

    # 注意：plev(lev) 語法錯誤，應該是 plev[lev]
    # 這裡的邏輯需要確保 prs 的維度與 tmk 匹配
    # 如果 tmk 是 4D (Time, pres_bottom_top, south_north, west_east)
    # 那麼 prs 也應該是 4D，並且每個層次的壓力值在空間上是常數
    
    # 更安全的方法是使用 xarray 的廣播功能來操作
    #prs_data = ds["pres_levels"].values * 100 # 將 hPa 轉換為 Pa
    # 將 plev 轉換為與 tmk 相同形狀的陣列
    # 例如，如果 tmk 是 (Time, Level, Lat, Lon)
    # 則 prs_expanded 應該是 (1, Level, 1, 1) 後再廣播
    
    # 這裡我們假設 prs 需要與 tmk 有相同的維度，並且每個水平點都有壓力值
    # 如果 ds["pres_levels"] 是一個 1D 的座標，那麼它需要被擴展到與 tmk 相同的形狀
    
    # 最佳的做法是 ds.coords["pres_levels"] 已經有對應的座標值，
    # 並且可以在 ds 中直接存取壓力變數 (例如 ds["P"] 或 ds["pressure"])。
    # 如果 'pres_levels' 只是索引，您需要從其他地方獲取實際的壓力場。
    # 這裡假設 ds["pres_levels"] 是壓力值本身，且可以在計算中被廣播。
    
    # 更正 prs 構造邏輯:
    # 假設 ds["pres_levels"] 是一個 DataArray 或可以直接作為 numpy 數組使用
    # prs 應該具有與 tmk 相同的形狀
    
    # 如果 ds["pres_levels"] 是 (pres_bottom_top,) 維度，則需要擴展
    # 假設 ds 包含一個 'pressure' 變數，其維度與 't' 相同
    # 或者，我們需要利用 ds["pres_levels"] 的座標來創建一個壓力場
    
    # 這裡假設 ds["plev"] 是一個具有正確維度 (pres_bottom_top) 的 DataArray
    # 並且可以直接用於廣播
    
    # 計算 w = -omega / (rho * g)
    # 或者使用 W = -omega * RT / (P * g)
    # 這裡使用 ds["w"].values * 287.05 * (tmk * (1 + 0.61 * qvp)) / prs / 9.81
    # 假設 ds["w"] 是 omega，並將其轉換為 w (m/s)
    
    # 如果 ds["pres_levels"] 是一個單獨的座標 DataArray，且與 'pres_bottom_top' 維度匹配：
    #prs_val = ds["pres_levels"] * 100 # hPa to Pa
    # 確保 prs_val 可以與 tmk 和 qvp 進行廣播
    # 如果 prs_val 是 (level,)，tmk 是 (Time, Level, Lat, Lon)，則會自動廣播
    plev = np.squeeze(ds["plev"].values) # hPa to Pa
    prs = np.ones(np.shape(tmk))
    for lv in range(len(plev)):
        prs[lv,:,:] = plev[lv]
    
    data = np.squeeze(ds["w"].values) * 287.05 * (tmk * (1 + 0.61 * qvp)) / prs / 9.81
    
    # 這裡假設 diag_wa_p 的輸出是 4D (Time, pres_bottom_top, south_north, west_east)
    return _create_dataarray(data, ds, "wa_p", "Vertical Velocity", "m/s")

def sat_vapor_pressure(T):  # T in Celsius
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))  # hPa

def diag_T2(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["2t"].values)
    return _create_dataarray(data, ds, "T2", "2m Temperature", "K")

def diag_Q2(ds: xr.Dataset) -> xr.DataArray:
    td_C = np.squeeze(ds["2d"].values) - 273.15
    e = sat_vapor_pressure(td_C) * 100 # Pa
    # 檢查 ds["sp"] 是否是表面壓力的正確變數名
    # q2 = 0.622 * e / (ds["sp"].values - e)
    # 如果 ds["sp"] 是 surface pressure (Pa)，這公式是正確的
    data = 0.622 * e / (np.squeeze(ds["sp"].values) - e)
    return _create_dataarray(data, ds, "Q2", "2m Mixing Ratio", "kg/kg")

def diag_rh2(ds: xr.Dataset) -> xr.DataArray:
    T2_C = np.squeeze(ds["2t"].values) - 273.15
    Td2_C = np.squeeze(ds["2d"].values) - 273.15
    e_sat = sat_vapor_pressure(T2_C) * 100
    e = sat_vapor_pressure(Td2_C) * 100
    data = e / e_sat * 100
    return _create_dataarray(data, ds, "rh2", "2m Relative Humidity", "%")

def diag_td2(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["2d"].values)
    return _create_dataarray(data, ds, "td2", "2m Dew Point Temperature", "K")

def diag_umet10(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["10u"].values)
    #print(np.shape(data), ds["10u"].dims)
    return _create_dataarray(data, ds, "umet10", "10m U-component of Wind", "m/s")

def diag_vmet10(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["10v"].values)
    return _create_dataarray(data, ds, "vmet10", "10m V-component of Wind", "m/s")

def diag_slp(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["msl"].values) / 100 # Pa to hPa
    return _create_dataarray(data, ds, "slp", "Sea Level Pressure", "hPa")

def diag_SST(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["sst"].values)
    return _create_dataarray(data, ds, "SST", "Sea Surface Temperature", "K")

def diag_PSFC(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["sp"].values)
    return _create_dataarray(data, ds, "PSFC", "Surface Pressure", "Pa")

def diag_pw(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["tcwv"].values)
    return _create_dataarray(data, ds, "pw", "Precipitable Water", "kg/m^2")

def diag_PBLH(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["blh"].values)
    return _create_dataarray(data, ds, "PBLH", "Planetary Boundary Layer Height", "m")

def diag_RAINNC(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["tp"].values)
    return _create_dataarray(data, ds, "RAINNC", "Total Precipitation", "m") # 單位需要確認是累積降水還是速率

def diag_SWDOWN(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["ssrd"].values)
    return _create_dataarray(data, ds, "SWDOWN", "Surface Shortwave Downward Radiation", "W/m^2") # 單位需要確認

def diag_OLR(ds: xr.Dataset) -> xr.DataArray:
    data = np.squeeze(ds["ttr"].values)
    return _create_dataarray(data, ds, "OLR", "Outgoing Longwave Radiation", "W/m^2") # 單位需要確認

def diag_REFL(ds: xr.Dataset) -> xr.DataArray:
    sn0 = 1
    ivarint = 1
    tmk = np.squeeze(ds["t"].values)
    qvp = np.squeeze(ds["q"].values) / (1 - np.squeeze(ds["q"].values))
    qra = np.squeeze(ds["crwc"].values) / (1 - np.squeeze(ds["crwc"].values))
    qsn = np.squeeze(ds["cswc"].values) / (1 - np.squeeze(ds["cswc"].values))
    qgr = np.squeeze(ds["q"].values) * 0 # qgr = 0

    qvp = np.maximum(qvp, 0)
    qra = np.maximum(qra, 0)
    qsn = np.maximum(qsn, 0)
    qgr = np.maximum(qgr, 0)

    if sn0 == 0:
        mask = tmk < 273.15
        qsn[mask] = qra[mask]
        qra[mask] = 0

    # 這裡的 prs 處理方式與 diag_wa_p 類似，需要確保與 tmk 維度兼容
    # 假設 ds["pres_levels"] 是壓力值 DataArray
    plev = np.squeeze(ds["plev"].values) # hPa to Pa
    prs = np.ones(np.shape(tmk))
    for lv in range(len(plev)):
        prs[lv,:,:] = plev[lv]
    
    # 再次確認 virtual_t 的計算公式，這裡應該是混合比 (mixing ratio)
    virtual_t = tmk * (1 + 0.61 * qvp)

    # 確保 rhoair 的計算使用正確維度的 prs 和 virtual_t
    rhoair = prs / (287.04 * virtual_t) # prs_val.values 假設可以自動廣播

    factor_r = 720 * 1e18 * (1 / (np.pi * 1000))**1.75
    factor_s = factor_r * (0.224 * (100 / 1000)**2)
    factor_g = factor_r * (0.224 * (400 / 1000)**2)

    z_e = (factor_r * (rhoair * qra)**1.75 / (8e6 if ivarint == 0 else 1e10)**0.75 +
           factor_s * (rhoair * qsn)**1.75 / (2e7 if ivarint == 0 else 2e8)**0.75 +
           factor_g * (rhoair * qgr)**1.75 / (4e6 if ivarint == 0 else 5e7)**0.75)

    dbz = 10 * np.log10(np.maximum(z_e, 0.001))
    
    # mxdbz = np.max(dbz, 1) 表示對第二個維度 (pres_bottom_top) 取最大值
    # 輸出將是 (Time, south_north, west_east)
    data = np.max(dbz, axis=0) # axis=1 通常代表第二個維度，這裡是指 pres_bottom_top
    
    # 這裡假設輸出是 3D，且時間維度是第一個維度
    return _create_dataarray(data, ds, "mREFL", "Emulated Radar Reflectivity", "dBZ")
