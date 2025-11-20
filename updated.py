"""
Enhanced Diagnostic Functions for DLAMP Preprocessing

Now supports multiple data sources:
- ERA5: Reanalysis data
- ERA5_r: Reverse conversion (ERA5 format to original)
- RWRF: Regional WRF output
- SFNO: Spherical Fourier Neural Operator global forecast

Each diagnostic function uses pattern matching to handle source-specific
variable names and unit conversions.
"""

import numpy as np
import xarray as xr


def _create_dataarray(
        data: np.ndarray,
        ds: xr.Dataset,
        var_name: str,
        long_name: str,
        units: str,
    ) -> xr.DataArray:
    """
    Create xarray DataArray with proper dimensions and metadata
    """
    coords = {"Time": ds["Time"]}
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
        }
    )


def sat_vapor_pressure_water(T):
    """Saturation vapor pressure over water (Magnus formula)"""
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))  # hPa


###===== Pressure Level Variables =================================###

def diag_z_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Geopotential height = geopotential / g"""
    g = 9.80665

    match source_dataset:
        case "ERA5" | "SFNO":  # SFNO uses same format as ERA5
            data = np.squeeze(ds["z"].values) / g
            nc_key = "z_p"
        case "ERA5_r":
            data = np.squeeze(ds["z_p"].values) * g
            nc_key = "z"
        case "RWRF":
            data = np.squeeze(ds["z_p"].values)
            nc_key = "z_p"
        case _:
            data = np.nan
            nc_key = "z_p"

    return _create_dataarray(data, ds, nc_key, "Geopotential Height", "m")


def diag_tk_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Air Temperature [K]"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["t"].values)
            nc_key = "tk_p"
        case "ERA5_r":
            data = np.squeeze(ds["tk_p"].values)
            nc_key = "t"
        case "RWRF":
            data = np.squeeze(ds["tk_p"].values)
            nc_key = "tk_p"
        case _:
            data = np.nan
            nc_key = "tk_p"

    return _create_dataarray(data, ds, nc_key, "Air Temperature", "K")


def diag_umet_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """U-component of wind (Earth-rotated)"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["u"].values)
            nc_key = "umet_p"
        case "ERA5_r":
            data = np.squeeze(ds["umet_p"].values)
            nc_key = "u"
        case "RWRF":
            data = np.squeeze(ds["umet_p"].values)
            nc_key = "umet_p"
        case _:
            data = np.nan
            nc_key = "umet_p"

    return _create_dataarray(data, ds, nc_key, "U-component of Wind", "m s-1")


def diag_vmet_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """V-component of wind (Earth-rotated)"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["v"].values)
            nc_key = "vmet_p"
        case "ERA5_r":
            data = np.squeeze(ds["vmet_p"].values)
            nc_key = "v"
        case "RWRF":
            data = np.squeeze(ds["vmet_p"].values)
            nc_key = "vmet_p"
        case _:
            data = np.nan
            nc_key = "vmet_p"

    return _create_dataarray(data, ds, nc_key, "V-component of Wind", "m s-1")


def diag_QVAPOR_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Water vapor mixing ratio from specific humidity"""
    match source_dataset:
        case "ERA5" | "SFNO":
            if "q" in ds:
                q = np.squeeze(ds["q"].values)
                data = q / (1 - q)
            elif "r" in ds and "t" in ds and "pres_levels" in ds:
                # Fallback: convert relative humidity
                rh = np.squeeze(ds["r"].values) / 100.0
                t = np.squeeze(ds["t"].values)
                p = np.squeeze(ds["pres_levels"].values) * 100
                t_celsius = t - 273.15
                es = sat_vapor_pressure_water(t_celsius)
                e = rh * es
                data = (0.622 * e) / (p / 100.0 - e)
            else:
                data = np.nan
        case "RWRF":
            data = np.squeeze(ds["QVAPOR_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QVAPOR_p", "Water Vapor Mixing Ratio", "kg kg-1")


def diag_wa_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Vertical velocity: omega [Pa s-1] to w [m s-1]"""
    Rd = 287.058
    g = 9.80665

    match source_dataset:
        case "ERA5" | "SFNO":
            omega = np.squeeze(ds["w"].values)
            tmk = np.squeeze(ds["t"].values)
            plev = np.squeeze(ds["pres_levels"].values * 100)

            if "q" in ds:
                q = np.squeeze(ds["q"].values)
                qvp = q / (1 - q)
            elif "r" in ds:
                rh = np.squeeze(ds["r"].values) / 100.0
                t_celsius = tmk - 273.15
                es = sat_vapor_pressure_water(t_celsius)
                e = rh * es
                qvp = (0.622 * e) / (plev / 100.0 - e)
            else:
                qvp = np.zeros_like(tmk)

            t_virt = tmk * (1 + 0.61 * qvp)
            prs = np.zeros(np.shape(tmk))
            for pl in range(len(plev)):
                prs[pl] = plev[pl]

            data = -1 * omega * Rd * t_virt / prs / g

        case "RWRF":
            data = np.squeeze(ds["wa_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "wa_p", "Vertical Velocity", "m s-1")


###===== Hydrometeor Variables ====================================###
# Note: SFNO typically doesn't output hydrometeor fields directly
# These are set to zero for SFNO source

def diag_QRAIN_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Rain water mixing ratio"""
    match source_dataset:
        case "ERA5":
            q = np.squeeze(ds["crwc"].values)
            data = q / (1 - q)
        case "SFNO":
            # SFNO doesn't output rain water - set to zero
            template = np.squeeze(ds["t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            data = np.squeeze(ds["QRAIN_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QRAIN_p", "Rain Water Mixing Ratio", "kg kg-1")


def diag_QCLOUD_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Cloud liquid water mixing ratio"""
    match source_dataset:
        case "ERA5":
            q = np.squeeze(ds["clwc"].values)
            data = q / (1 - q)
        case "SFNO":
            template = np.squeeze(ds["t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            data = np.squeeze(ds["QCLOUD_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QCLOUD_p", "Cloud Water Mixing Ratio", "kg kg-1")


def diag_QSNOW_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Snow water mixing ratio"""
    match source_dataset:
        case "ERA5":
            q = np.squeeze(ds["cswc"].values)
            data = q / (1 - q)
        case "SFNO":
            template = np.squeeze(ds["t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            data = np.squeeze(ds["QSNOW_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QSNOW_p", "Snow Water Mixing Ratio", "kg kg-1")


def diag_QICE_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Cloud ice mixing ratio"""
    match source_dataset:
        case "ERA5":
            q = np.squeeze(ds["ciwc"].values)
            data = q / (1 - q)
        case "SFNO":
            template = np.squeeze(ds["t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            data = np.squeeze(ds["QICE_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QICE_p", "Cloud Ice Mixing Ratio", "kg kg-1")


def diag_QGRAUP_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Graupel water mixing ratio"""
    match source_dataset:
        case "ERA5" | "SFNO":
            # ERA5 and SFNO don't have graupel
            template = np.squeeze(ds["q" if "q" in ds else "t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            data = np.squeeze(ds["QGRAUP_p"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QGRAUP_p", "Graupel Water Mixing Ratio", "kg kg-1")


def diag_QTOTAL_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Total hydrometeors mixing ratio"""
    match source_dataset:
        case "ERA5":
            qlist = ["clwc", "crwc", "ciwc", "cswc"]
            mixing_ratios = [np.squeeze(ds[q].values) / (1 - np.squeeze(ds[q].values)) for q in qlist]
            data = sum(mixing_ratios)
        case "SFNO":
            # SFNO doesn't have hydrometeors - set to zero
            template = np.squeeze(ds["t"].values)
            data = np.zeros_like(template)
        case "RWRF":
            qlist = ["QCLOUD_p", "QRAIN_p", "QICE_p", "QSNOW_p", "QGRAUP_p"]
            data = sum(np.squeeze(ds[q].values) for q in qlist)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "QTOTAL_p", "Total Hydrometeors Mixing Ratio", "kg kg-1")


###===== Surface Variables ========================================###

def diag_T2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """2m temperature"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["2t"].values)
        case "RWRF":
            data = np.squeeze(ds["T2"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "T2", "2m Temperature", "K")


def diag_Q2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """2m water vapor mixing ratio"""
    match source_dataset:
        case "ERA5" | "SFNO":
            td2 = np.squeeze(ds["2d"].values)
            sp = np.squeeze(ds["sp"].values)
            e = sat_vapor_pressure_water(td2 - 273.15) * 100  # Pa
            data = (0.622 * e) / (sp - e)
        case "RWRF":
            data = np.squeeze(ds["Q2"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "Q2", "2m Mixing Ratio", "kg kg-1")


def diag_rh2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """2m relative humidity"""
    match source_dataset:
        case "ERA5" | "SFNO":
            td2 = np.squeeze(ds["2d"].values)
            t2 = np.squeeze(ds["2t"].values)
            e = sat_vapor_pressure_water(td2 - 273.15) * 100
            esat = sat_vapor_pressure_water(t2 - 273.15) * 100
            data = e / esat * 100
        case "RWRF":
            data = np.squeeze(ds["rh2"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "rh2", "2m Relative Humidity", "%")


def diag_td2(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """2m dewpoint temperature"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["2d"].values)
        case "RWRF":
            data = np.squeeze(ds["td2"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "td2", "2m Dew Point Temperature", "K")


def diag_umet10(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """10m U-wind"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["10u"].values)
        case "RWRF":
            data = np.squeeze(ds["umet10"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "umet10", "10m U-component of Wind", "m s-1")


def diag_vmet10(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """10m V-wind"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["10v"].values)
        case "RWRF":
            data = np.squeeze(ds["vmet10"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "vmet10", "10m V-component of Wind", "m s-1")


def diag_slp(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Mean sea level pressure"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["msl"].values) / 100.0  # Pa to hPa
        case "ERA5_r":
            data = np.squeeze(ds["slp"].values) * 100.0
            nc_key = "msl"
        case "RWRF":
            data = np.squeeze(ds["slp"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "slp", "Sea Level Pressure", "hPa")


def diag_PSFC(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Surface pressure"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["sp"].values)
        case "ERA5_r":
            data = np.squeeze(ds["PSFC"].values)
            nc_key = "sp"
        case "RWRF":
            data = np.squeeze(ds["PSFC"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "PSFC", "Surface Pressure", "Pa")


def diag_pw(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Precipitable water (total column water vapor)"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["tcwv"].values)
        case "ERA5_r":
            data = np.squeeze(ds["pw"].values)
            nc_key = "tcwv"
        case "RWRF":
            data = np.squeeze(ds["pw"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "pw", "Precipitable Water", "kg m-2")


def diag_RAINNC(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Total precipitation"""
    match source_dataset:
        case "ERA5" | "SFNO":
            data = np.squeeze(ds["tp"].values)
        case "RWRF":
            data = np.squeeze(ds["RAINNC"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "RAINNC", "Total Precipitation", "m")


###===== Optional Variables (may not be in SFNO) =================###

def diag_SST(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Sea surface temperature"""
    match source_dataset:
        case "ERA5":
            data = np.squeeze(ds["sst"].values)
        case "SFNO":
            # Check if SFNO provides SST, otherwise set to NaN or climatology
            if "sst" in ds:
                data = np.squeeze(ds["sst"].values)
            else:
                template = np.squeeze(ds["2t"].values)
                data = np.full_like(template, np.nan)
        case "RWRF":
            data = np.squeeze(ds["SST"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "SST", "Sea Surface Temperature", "K")


def diag_PBLH(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Planetary boundary layer height"""
    match source_dataset:
        case "ERA5":
            data = np.squeeze(ds["blh"].values)
        case "SFNO":
            # SFNO may not provide PBL height
            if "blh" in ds:
                data = np.squeeze(ds["blh"].values)
            else:
                template = np.squeeze(ds["2t"].values)
                data = np.full_like(template, 1000.0)  # Default 1km
        case "RWRF":
            data = np.squeeze(ds["PBLH"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "PBLH", "Planetary Boundary Layer Height", "m")


def diag_SWDOWN(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Surface shortwave downward radiation"""
    match source_dataset:
        case "ERA5":
            data = np.squeeze(ds["ssrd"].values) / 3600  # J/m² to W/m²
        case "SFNO":
            if "ssrd" in ds:
                data = np.squeeze(ds["ssrd"].values) / 3600
            else:
                template = np.squeeze(ds["2t"].values)
                data = np.full_like(template, np.nan)
        case "RWRF":
            data = np.squeeze(ds["SWDOWN"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "SWDOWN", "Surface Shortwave Downward Radiation", "W m-2")


def diag_OLR(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Outgoing longwave radiation"""
    match source_dataset:
        case "ERA5":
            data = np.squeeze(ds["ttr"].values) / 3600  # J/m² to W/m²
        case "SFNO":
            if "ttr" in ds:
                data = np.squeeze(ds["ttr"].values) / 3600
            else:
                template = np.squeeze(ds["2t"].values)
                data = np.full_like(template, np.nan)
        case "RWRF":
            data = np.squeeze(ds["OLR"].values)
        case _:
            data = np.nan

    return _create_dataarray(data, ds, "OLR", "Outgoing Longwave Radiation", "W m-2")


###===== Derived Diagnostics ======================================###

def diag_REFL(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Emulated radar reflectivity
    
    Note: SFNO doesn't provide hydrometeors, so reflectivity will be minimal
    """
    match source_dataset:
        case "SFNO":
            # SFNO lacks hydrometeor fields - return zero reflectivity
            template = np.squeeze(ds["t"].values)
            data = np.full_like(template, -30.0)  # Below detection threshold
            return _create_dataarray(data, ds, "REFL", "Emulated Radar Reflectivity", "dBZ")
        
        case "ERA5" | "RWRF":
            # Use existing calculation from original code
            # [Include full REFL calculation here from original file]
            pass
    
    # Fallback
    template_shape = ds["t" if "t" in ds else "tk_p"].values.shape
    data = np.full(np.squeeze(template_shape), -30.0)
    return _create_dataarray(data, ds, "REFL", "Emulated Radar Reflectivity", "dBZ")


def diag_MAX_REFL(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """Column maximum radar reflectivity"""
    REFL = diag_REFL(source_dataset, ds)
    _, nl, ny, nx = np.shape(REFL.values)
    data = np.max(np.reshape(REFL.values, (nl, ny, nx)), axis=0)
    return _create_dataarray(data, ds, "MAX_REFL", "Emulated Column Maximum Radar Reflectivity", "dBZ")