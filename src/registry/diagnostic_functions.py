import numpy as np
import xarray as xr
import yaml

def _create_dataarray(
        data: np.ndarray, 
        ds: xr.Dataset, 
        var_name: str, 
        long_name: str, 
        units: str,
    ) -> xr.DataArray:
    """
    
    """

    # setup default coords
    coords = {"Time": ds["Time"]}
    #coords["pres_bottom_top"] = {}
    dims = ["Time"]

    if var_name == "pres_levels":
        dims.append("pres_bottom_top")
        coords["pres_bottom_top"] = ds["pres_bottom_top"]

    elif var_name in ["XLONG_C", "XLAT_C"]:
        dims += ["corner_south_north", "corner_west_east"]
        coords["corner_south_north"] = ("corner_south_north", np.linspace(-450, 450, 451))
        coords["corner_west_east"] = ("corner_west_east", np.linspace(-450, 450, 451))

    elif data.ndim == 3:
        dims += ["pres_bottom_top", "south_north", "west_east"]
        coords.update({
            "pres_bottom_top": ds["pres_bottom_top"],
            "south_north": ds["south_north"],
            "west_east": ds["west_east"],
        })

    elif data.ndim == 2:
        dims += ["south_north", "west_east"]
        coords.update({
            "south_north": ds["south_north"],
            "west_east": ds["west_east"],
        })

    return xr.DataArray(
        np.expand_dims(data, axis=0).astype(np.float32),
        coords=coords,
        dims=dims,
        name=var_name,
        attrs={
            "long_name": long_name,
            "units": units,
            #"dtype": str(data.dtype),
        }
    )


def diag_z_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Geopotential height = geopotential / g
    
    """
    g = 9.80665 # Standard gravity constant

    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["z"].values) / g
            
        case "RWRF":
            data = np.squeeze(ds["z_p"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "z_p", "Geopotential Height", "m")


def diag_tk_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Air Temperature [K]
    
    """

    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["t"].values)
            
        case "RWRF":
            data = np.squeeze(ds["tk_p"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "tk_p", "Air Temperature", "K")


def diag_umet_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    U-component of wind (Earth-rotated)
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["u"].values)
            
        case "RWRF":
            data = np.squeeze(ds["umet_p"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "umet_p", "U-component of Wind", "m s-1")


def diag_vmet_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    V-component of wind (Earth-rotated)
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["v"].values)
            
        case "RWRF":
            data = np.squeeze(ds["vmet_p"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "vmet_p", "V-component of Wind", "m s-1")


def diag_QVAPOR_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific humidity to water vapor mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["q"].values)
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QVAPOR_p"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "QVAPOR_p", "Water Vapor Mixing Ratio", "kg kg-1")


def diag_QRAIN_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific rain water content to rain water mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["crwc"].values)
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QRAIN_p"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QRAIN_p", "Rain Water Mixing Ratio", "kg kg-1")


def diag_QCLOUD_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific cloud liquid water content to cloud water mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["clwc"].values)
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QCLOUD_p"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "QCLOUD_p", "Cloud Water Mixing Ratio", "kg kg-1")


def diag_QSNOW_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific snow water content to snow water mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["cswc"].values)
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QSNOW_p"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "QSNOW_p", "Snow Water Mixing Ratio", "kg kg-1")


def diag_QICE_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific cloud ice content to cloud ice mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["ciwc"].values)
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QICE_p"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "QICE_p", "Cloud Ice Mixing Ratio", "kg kg-1")


def diag_QGRAUP_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific graupel water content to graupel water mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            q = np.squeeze(ds["q"].values) * 0
            data = q/(1-q)
            
        case "RWRF":
            data = np.squeeze(ds["QGRAUP_p"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "QGRAUP_p", "Graupel Water Mixing Ratio", "kg kg-1")


def diag_QWTAER_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Total hydrometeors mixing ratio
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = 0
            qlist = ["clwc", "crwc", "ciwc", "cswc"]
            for q in qlist:
                data += (ds[q].values / (1 - ds[q].values))
            
        case "RWRF":
            qlist = ["QCLOUD_p", "QRAIN_p", "QICE_p", "QSNOW_p", "QGRAUP_p"]
            for q in qlist:
                data += ds[q].values
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "QWATER_p", "Total Hydrometeors Mixing Ratio", "kg kg-1")


def diag_wa_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    omega [Pa s-1] to w [m s-1]
    
    """
    Rd = 287.058 # Dry air gas constant J/(kg K)
    epsilon = 0.622 # Ratio of molecular weight of water vapor to dry air
    g = 9.80665 # Standard gravity constant
    
    match source_dataset:
        
        case "ERA5":
            omega = np.squeeze(ds["w"].values)
            tmk = np.squeeze(ds["t"].values)
            q = np.squeeze(ds["q"].values)
            qvp = q/(1-q)
            plev = np.squeeze(ds["pres_levels"].values * 100)
            t_virt = tmk * (1 + 0.61 * qvp)
            
            prs = np.zeros(np.shape(tmk))
            for pl in range(len(plev)):
                prs[pl] = plev[pl]
                
            #data1 = -1 * omega * Rd * t_virt / prs / g
            data = -1 * omega * Rd * tmk / prs / g
            
        case "RWRF":
            data = np.squeeze(ds["wa_p"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "wa_p", "Vertical Velocity", "m s-1")


def sat_vapor_pressure_water(T):  # T in Celsius
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))  # hPa


def diag_T2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Air Temperature at 2 m height above surface 
    
    """ 
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["2t"].values)
            
        case "RWRF":
            data = np.squeeze(ds["T2"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "T2", "2m Temperature", "K")


def diag_Q2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    water vapor mixing ratio at 2 m height above surface
    
    """

    match source_dataset:
        
        case "ERA5":
            td2 = np.squeeze(ds["2d"].values)
            sp = np.squeeze(ds["sp"].values)
            e = sat_vapor_pressure_water(td2 - 273.15) * 100  # [Pa]
            data = (0.622 * e) / (sp - e)
            
        case "RWRF":
            data = np.squeeze(ds["Q2"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "Q2", "2m Mixing Ratio", "kg kg-1")


def diag_rh2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    relative humidity at 2 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            td2 = np.squeeze(ds["2d"].values)
            t2 = np.squeeze(ds["2t"].values)
            sp = np.squeeze(ds["sp"].values)
            e = sat_vapor_pressure_water(td2 - 273.15) * 100  # [Pa]
            esat = sat_vapor_pressure_water(t2 - 273.15) * 100  # [Pa]
            data = e / esat * 100
            
        case "RWRF":
            data = np.squeeze(ds["rh2"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "rh2", "2m Relative Humidity", "%")


def diag_td2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    dew-point temperature at 2 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["2d"].values)
            
        case "RWRF":
            data = np.squeeze(ds["td2"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "td2", "2m Dew Point Temperature", "K")


def diag_umet10(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    U-wind at 10 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["10u"].values)
            
        case "RWRF":
            data = np.squeeze(ds["umet10"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "umet10", "10m U-component of Wind", "m s-1")


def diag_vmet10(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    V-wind at 10 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["10v"].values)
            
        case "RWRF":
            data = np.squeeze(ds["vmet10"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "vmet10", "10m V-component of Wind", "m s-1")


def diag_slp(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    sea-level pressure at surface [hPa]
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["msl"].values)
            
        case "RWRF":
            data = np.squeeze(ds["slp"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "slp", "Sea Level Pressure", "hPa")


def diag_SST(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    sea surface temperature
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["sst"].values)
            
        case "RWRF":
            data = np.squeeze(ds["SST"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "SST", "Sea Surface Temperature", "K")


def diag_PSFC(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    surface pressure [Pa]
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["sp"].values)
            
        case "RWRF":
            data = np.squeeze(ds["PSFC"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "PSFC", "Surface Pressure", "Pa")


def diag_pw(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    dew-point temperature at 2 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["tcwv"].values)
            
        case "RWRF":
            data = np.squeeze(ds["pw"].values)
            
        case _:
            data = np.nan
    
    return _create_dataarray(data, ds, "pw", "Precipitable Water", "kg m-2")


def diag_PBLH(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    dew-point temperature at 2 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["blh"].values)
            
        case "RWRF":
            data = np.squeeze(ds["PBLH"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "PBLH", "Planetary Boundary Layer Height", "m")


def diag_RAINNC(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    Total Precipitation
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["tp"].values)
            
        case "RWRF":
            data = np.squeeze(ds["RAINNC"].values)
            
        case _:
            data = np.nan
            
    return _create_dataarray(data, ds, "RAINNC", "Total Precipitation", "m") 


def diag_SWDOWN(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    Downward shortwave radiation at surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["ssrd"].values) / 3600
            
        case "RWRF":
            data = np.squeeze(ds["SWDOWN"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "SWDOWN", "Surface Shortwave Downward Radiation", "W m-2") 


def diag_OLR(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    Outward longwave radiation at TOA
    
    """
    
    match source_dataset:
        
        case "ERA5":
            data = np.squeeze(ds["ttr"].values) / 3600
            
        case "RWRF":
            data = np.squeeze(ds["OLR"].values)
            
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "OLR", "Outgoing Longwave Radiation", "W m-2") 


def diag_mREFL(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """ 
    dew-point temperature at 2 m height above surface
    
    """
    
    match source_dataset:
        
        case "ERA5":
            tmk = np.squeeze(ds["t"].values)
            qvp = np.squeeze(ds["q"].values)/(1-np.squeeze(ds["q"].values))
            qra = np.squeeze(ds["clwc"].values/(1-ds["clwc"].values))
            qsn = np.squeeze(ds["cswc"].values/(1-ds["cswc"].values))
            qgr = np.zeros(np.shape(tmk))
            prs = np.zeros(np.shape(tmk))
            plev = np.squeeze(ds["pres_levels"].values)
            for pl in range(len(plev)):
                prs[pl,:,:] = (plev[pl] * 100)
            
        case "RWRF":
            tmk = np.squeeze(ds["tk_p"].values)
            qvp = np.squeeze(ds["QVAPOR_p"].values)
            qra = np.squeeze(ds["QRAIN_p"].values)
            qsn = np.squeeze(ds["QSNOW_p"].values)
            qgr = np.squeeze(ds["QGRAUP_p"].values)
            prs = np.zeros(np.shape(tmk))
            plev = np.squeeze(ds["pres_levels"].values)
            for pl in range(len(plev)):
                prs[pl,:,:] = (plev[pl] * 100.0)
            
        case _:
            data = np.nan
            
    sn0 = 1
    ivarint = 1
    qvp = np.maximum(qvp, 0)
    qra = np.maximum(qra, 0)
    qsn = np.maximum(qsn, 0)
    qgr = np.maximum(qgr, 0)

    if sn0 == 1:
        mask = tmk < 273.15
        qsn[mask] = qra[mask]
        qra[mask] = 0
    
    virtual_t = tmk * (1 + 0.61 * qvp)

    rhoair = prs / (287.04 * virtual_t) # prs_val.values 假設可以自動廣播

    factor_r = 720 * 1e18 * (1 / (np.pi * 1000))**1.75
    factor_s = factor_r * (0.224 * (100 / 1000)**2)
    factor_g = factor_r * (0.224 * (400 / 1000)**2)

    z_e = (factor_r * (rhoair * qra)**1.75 / (8e6 if ivarint == 0 else 1e10)**0.75 +
           factor_s * (rhoair * qsn)**1.75 / (2e7 if ivarint == 0 else 2e8)**0.75 +
           factor_g * (rhoair * qgr)**1.75 / (4e6 if ivarint == 0 else 5e7)**0.75)

    dbz = 10 * np.log10(np.maximum(z_e, 0.001))
    
    data = np.max(dbz, axis=0)
    
    return _create_dataarray(data, ds, "mREFL", "Emulated Column Maximum Radar Reflectivity", "dBZ")
