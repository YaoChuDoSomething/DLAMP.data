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

def sat_vapor_pressure_water(T):  # T in Celsius
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))  # hPa

def diag_z_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Geopotential height = geopotential / g

    """
    g = 9.80665 # Standard gravity constant

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["z"].values) / g
            nc_key = "z_p"

        case "ERA5_r":
            data = np.squeeze(ds["z_p"].values) * g
            nc_key = "z"

        case "RWRF":
            data = np.squeeze(ds["z_p"].values)

        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "Geopotential Height", "m")


def diag_tk_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Air Temperature [K]

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["t"].values)
            nc_key = "tk_p"

        case "ERA5_r":
            data = np.squeeze(ds["tk_p"].values)
            nc_key = "t"

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
            nc_key = "umet_p"

        case "ERA5_r":
            data = np.squeeze(ds["umet_p"].values)
            nc_key = "u"

        case "RWRF":
            data = np.squeeze(ds["umet_p"].values)
            nc_key = "umet_p"

        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "U-component of Wind", "m s-1")


def diag_vmet_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    V-component of wind (Earth-rotated)

    """

    match source_dataset:

        case "ERA5":
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

    return _create_dataarray(data, ds, nc_key, "V-component of Wind", "m s-1")


def diag_QVAPOR_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Specific humidity to water vapor mixing ratio

    """

    match source_dataset:

        case "ERA5":
            if "q" in ds:
                q = np.squeeze(ds["q"].values)
                data = q/(1-q)
            elif "r" in ds and "t" in ds and "pres_levels" in ds:
                # Convert relative humidity to mixing ratio
                rh = np.squeeze(ds["r"].values) / 100.0  # convert to fraction
                t = np.squeeze(ds["t"].values)  # temperature in Kelvin
                p = np.squeeze(ds["pres_levels"].values) * 100 # pressure in Pa

                # Convert temperature to Celsius for vapor pressure calculation
                t_celsius = t - 273.15

                # Saturation vapor pressure (hPa)
                es = sat_vapor_pressure_water(t_celsius)

                # Actual vapor pressure (hPa)
                e = rh * es

                # Mixing ratio
                data = (0.622 * e) / (p / 100.0 - e) # p/100.0 to convert Pa to hPa
            else:
                data = np.nan

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


def diag_QTOTAL_p(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Total hydrometeors mixing ratio

    """

    match source_dataset:

        case "ERA5":
            qlist = ["clwc", "crwc", "ciwc", "cswc"]
            # Convert each specific humidity to mixing ratio first, then sum them up.
            mixing_ratios = [np.squeeze(ds[q].values) / (1 - np.squeeze(ds[q].values)) for q in qlist]
            data = sum(mixing_ratios)
            nc_key = "QTOTAL_p"

        case "RWRF":
            qlist = ["QCLOUD_p", "QRAIN_p", "QICE_p", "QSNOW_p", "QGRAUP_p"]
            data = sum(np.squeeze(ds[q].values) for q in qlist)
            nc_key = "QTOTAL_p"
        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "Total Hydrometeors Mixing Ratio", "kg kg-1")


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
            plev = np.squeeze(ds["pres_levels"].values * 100) # pressure in Pa

            if "q" in ds:
                q = np.squeeze(ds["q"].values)
                qvp = q / (1 - q)
            elif "r" in ds and "t" in ds and "pres_levels" in ds:
                # Convert relative humidity to mixing ratio
                rh = np.squeeze(ds["r"].values) / 100.0  # convert to fraction
                t_celsius = tmk - 273.15 # temperature in Celsius

                # Saturation vapor pressure (hPa)
                es = sat_vapor_pressure_water(t_celsius)

                # Actual vapor pressure (hPa)
                e = rh * es

                # Mixing ratio
                qvp = (0.622 * e) / (plev / 100.0 - e) # plev/100.0 to convert Pa to hPa
            else:
                qvp = np.zeros_like(tmk) # Default to 0 if no humidity data

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

def diag_umet100(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    U-wind at 100 m height above surface

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["100u"].values)
            nc_key = "umet100"

        case "ERA5_r":
            data = np.squeeze(ds["umet100"].values)
            nc_key = "u100m"

        case "RWRF":
            data = np.squeeze(ds["umet100"].values)
            nc_key = "umet100"

        case _:
            data = np.nan
            nc_key = "umet100"

    return _create_dataarray(data, ds, nc_key, "100m U-component of Wind", "m s-1")


def diag_vmet100(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    V-wind at 100 m height above surface

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["100v"].values)
            nc_key = "vmet100"

        case "ERA5_r":
            data = np.squeeze(ds["vmet100"].values)
            nc_key = "v100m"

        case "RWRF":
            data = np.squeeze(ds["vmet100"].values)
            nc_key = "vmet100"

        case _:
            data = np.nan
            nc_key = "vmet100"

    return _create_dataarray(data, ds, nc_key, "100m V-component of Wind", "m s-1")


def diag_slp(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    sea-level pressure at surface [hPa]

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["msl"].values) / 100.
            nc_key = "slp"

        case "ERA5_r":
            data = np.squeeze(ds["slp"].values) * 100.
            nc_key = "msl"

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
            nc_key = "SST"
            #sst[np.isnan(sst)] = np.nanmean(sst.ravel())
            #data = sst

        case "ERA5_r":
            data = np.squeeze(ds["SST"].values)
            nc_key = "sst"

        case "RWRF":
            data = np.squeeze(ds["SST"].values)

        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "Sea Surface Temperature", "K")


def diag_PSFC(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    surface pressure [Pa]

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["sp"].values)
            nc_key = "PSFC"

        case "ERA5_r":
            data = np.squeeze(ds["PSFC"].values)
            nc_key = "sp"

        case "RWRF":
            data = np.squeeze(ds["PSFC"].values)
            nc_key = "PSFC"

        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "Surface Pressure", "Pa")


def diag_pw(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    dew-point temperature at 2 m height above surface

    """

    match source_dataset:

        case "ERA5":
            data = np.squeeze(ds["tcwv"].values)
            nc_key = "pw"

        case "ERA5_r":
            data = np.squeeze(ds["pw"].values)
            nc_key = "tcwv"

        case "RWRF":
            data = np.squeeze(ds["pw"].values)
            nc_key = "pw"

        case _:
            data = np.nan

    return _create_dataarray(data, ds, nc_key, "Precipitable Water", "kg m-2")


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


def diag_REFL(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Emulated Radar Reflectivity

    """

    match source_dataset:

        case "ERA5":
            tmk = np.squeeze(ds["t"].values)
            qvp = np.squeeze(ds["q"].values)/(1-np.squeeze(ds["q"].values))
            qra = np.squeeze(ds["crwc"].values/(1-ds["crwc"].values))
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
            template_shape = ds["t" if "t" in ds else "tk_p"].values.shape
            data = np.full(np.squeeze(template_shape), np.nan)
            return _create_dataarray(data, ds, "REFL", "Emulated Radar Reflectivity", "dBZ")

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

    data = 10 * np.log10(np.maximum(z_e, 0.001))

    return _create_dataarray(data, ds, "REFL", "Emulated Radar Reflectivity", "dBZ")

def diag_MAX_REFL(source_dataset: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Emulated Column Maximum Radar Reflectivity
    """
    REFL = diag_REFL(source_dataset, ds)
    # The dimensions of REFL.values are (Time, pres_bottom_top, south_north, west_east)
    # We take the maximum over the pressure level axis (axis=1)
    _, nl, ny, nx = np.shape(REFL.values)

    data =  np.max(np.reshape(REFL.values, (nl, ny, nx)), axis=0)

    return _create_dataarray(data, ds, "MAX_REFL", "Emulated Column Maximum Radar Reflectivity", "dBZ")

