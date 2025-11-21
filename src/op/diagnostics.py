import numpy as np
import xarray as xr
from typing import Optional, List
from dataclasses import dataclass
from .utils.file_utils import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------------------
# 1. Constants & Physics Core (物理核心)
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetConstants:
    """Meteorological Constants."""

    g: float = 9.80665  # Gravity [m s-2]
    Rd: float = 287.058  # Gas constant for dry air [J kg-1 K-1]
    Rv: float = 461.5  # Gas constant for water vapor [J kg-1 K-1]
    epsilon: float = 0.622  # Ratio of molecular weights (Rd/Rv)
    T0: float = 273.15  # Zero Celsius in Kelvin


class Thermodynamics:
    """
    Pure physics calculations.
    Operates on numpy arrays, independent of xarray structures.
    """

    @staticmethod
    def sat_vapor_pressure_water(temp_k: np.ndarray) -> np.ndarray:
        """Magnus formula for saturation vapor pressure over water [Pa]."""
        # Input Temp in Kelvin, Formula uses Celsius
        t_c = temp_k - MetConstants.T0
        # Result in hPa -> convert to Pa (* 100)
        es_hpa = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        return es_hpa * 100.0

    @staticmethod
    def specific_humidity_to_mixing_ratio(q: np.ndarray) -> np.ndarray:
        """Convert specific humidity [kg/kg] to mixing ratio [kg/kg]."""
        # w = q / (1 - q)
        return q / (1.0 - q)

    @staticmethod
    def relative_humidity_to_mixing_ratio(
        rh_percent: np.ndarray, temp_k: np.ndarray, pressure_pa: np.ndarray
    ) -> np.ndarray:
        """Convert RH [%] to mixing ratio [kg/kg]."""
        es = Thermodynamics.sat_vapor_pressure_water(temp_k)
        e = (rh_percent / 100.0) * es
        # w = epsilon * e / (p - e)
        w = (MetConstants.epsilon * e) / (pressure_pa - e)
        return w

    @staticmethod
    def omega_to_w(
        omega: np.ndarray,
        temp_k: np.ndarray,
        pressure_pa: np.ndarray,
        q_mixing_ratio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert Omega [Pa/s] to Vertical Velocity w [m/s] using hydrostatic approximation.
        w = - omega / (rho * g) ~= - (omega * R * Tv) / (p * g)
        """
        # Calculate Virtual Temperature
        if q_mixing_ratio is not None:
            tv = temp_k * (1.0 + 0.61 * q_mixing_ratio)
        else:
            tv = temp_k

        w = -1.0 * omega * MetConstants.Rd * tv / (pressure_pa * MetConstants.g)
        return w

    @staticmethod
    def dewpoint_to_mixing_ratio(
        td_k: np.ndarray, pressure_pa: np.ndarray
    ) -> np.ndarray:
        """Convert Dewpoint [K] to Mixing Ratio [kg/kg]."""
        e = Thermodynamics.sat_vapor_pressure_water(td_k)
        w = (MetConstants.epsilon * e) / (pressure_pa - e)
        return w


# ------------------------------------------------------------------------------
# 2. Data Builder (資料封裝與標準化)
# ------------------------------------------------------------------------------


class DataBuilder:
    """
    Responsible for creating standardized xarray DataArrays.
    Handles dimension mapping and attribute assignment.
    """

    @staticmethod
    def build(
        data: np.ndarray,
        template_ds: xr.Dataset,
        name: str,
        description: str,
        units: str,
        is_3d: bool = True,
    ) -> xr.DataArray:
        # 1. Determine Coordinates based on input shape and template
        coords = {}
        dims = []

        # Time is always first
        if "Time" in template_ds.coords:
            dims.append("Time")
            coords["Time"] = template_ds["Time"]
        elif "time" in template_ds.coords:
            dims.append("time")
            coords["time"] = template_ds["time"]

        # Vertical Dimension (if 3D)
        if is_3d:
            # Try standard names
            v_dims = ["pres_bottom_top", "level", "pressure_level", "pres_levels"]
            found_v = next((d for d in v_dims if d in template_ds.coords), None)
            if found_v:
                dims.append(found_v)
                coords[found_v] = template_ds[found_v]

        # Horizontal Dimensions
        # Case A: Lat/Lon Grid (Global/ERA5)
        if "lat" in template_ds.coords and "lon" in template_ds.coords:
            dims.extend(["lat", "lon"])
            coords["lat"] = template_ds["lat"]
            coords["lon"] = template_ds["lon"]
        # Case B: WRF/Regional Grid (south_north, west_east)
        elif "south_north" in template_ds.dims and "west_east" in template_ds.dims:
            dims.extend(["south_north", "west_east"])
            # Preserve coordinates if they exist (e.g., XLAT, XLONG)
            for coord_name in ["XLAT", "XLONG", "XLAT_M", "XLONG_M"]:
                if coord_name in template_ds:
                    coords[coord_name] = template_ds[coord_name]

        # 2. Create DataArray
        # Ensure data matches dimensions (simple check)
        # Note: In a real scenario, strict shape checking is needed.

        # Expand dims if data is missing time dimension but dataset has it
        if len(dims) > data.ndim:
            data = np.expand_dims(data, axis=0)

        return xr.DataArray(
            data.astype(np.float32),
            coords=coords,
            dims=dims,
            name=name,
            attrs={
                "long_name": description,
                "units": units,
            },
        )


# ------------------------------------------------------------------------------
# 3. Diagnostic Engine (診斷變數計算器)
# ------------------------------------------------------------------------------


class DiagnosticEngine:
    """
    Main engine to derive diagnostic variables.
    Automatically detects available variables in the source dataset to determine
    the calculation strategy.
    """

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def _get_var(self, possible_names: List[str]) -> Optional[np.ndarray]:
        """Helper to get raw numpy array from the first matching variable name."""
        for name in possible_names:
            if name in self.ds:
                return np.squeeze(self.ds[name].values)
        return None

    def _get_pressure_pa(self) -> np.ndarray:
        """Derive 3D pressure field in Pascals."""
        # Strategy 1: Explicit 3D pressure variable
        pres = self._get_var(["pressure", "pres", "p"])
        if pres is not None:
            return pres

        # Strategy 2: Pressure levels (1D) broadcasted
        # Find the pressure dimension
        plev_names = ["level", "pres_levels", "pressure_level", "isobaricInhPa"]
        for p_name in plev_names:
            if p_name in self.ds.coords:
                plevs = self.ds[p_name].values
                # Heuristic: if max < 2000, assume hPa -> convert to Pa
                if np.max(plevs) < 2000:
                    plevs = plevs * 100.0

                # Broadcast to 3D/4D shape matching Temperature
                t_shape = self.ds[
                    next(v for v in ["t", "tk_p", "T"] if v in self.ds)
                ].shape
                # Create an array of shape t_shape where the vertical axis is plevs
                # Assuming (Time, Level, Lat, Lon) or (Level, Lat, Lon)
                # This is a simplification; strict broadcasting logic depends on dims

                # Robust broadcasting using xarray then extracting values
                p_da = xr.DataArray(plevs, coords={p_name: plevs}, dims=p_name)
                t_da = self.ds[next(v for v in ["t", "tk_p", "T"] if v in self.ds)]
                p_broadcasted, _ = xr.broadcast(p_da, t_da)
                return p_broadcasted.values

        raise ValueError("Could not determine pressure field from dataset.")

    # ===== Pressure Level Diagnostics =====

    def calc_geopotential_height(self) -> xr.DataArray:
        """Output: z_p [m]"""
        # Strategy: Z = z/g (ERA5) or Z=z_p (WRF)
        z_raw = self._get_var(["z", "geopotential"])
        if z_raw is not None:
            data = z_raw / MetConstants.g
        else:
            z_p_raw = self._get_var(["z_p", "GHT"])
            if z_p_raw is not None:
                data = z_p_raw
            else:
                raise ValueError(
                    "Missing source variable for Geopotential Height (z or z_p)"
                )

        return DataBuilder.build(data, self.ds, "z_p", "Geopotential Height", "m")

    def calc_temperature(self) -> xr.DataArray:
        """Output: tk_p [K]"""
        data = self._get_var(["t", "tk_p", "T", "temperature"])
        if data is None:
            raise ValueError("Missing source variable for Temperature")
        return DataBuilder.build(data, self.ds, "tk_p", "Air Temperature", "K")

    def calc_u_wind(self) -> xr.DataArray:
        """Output: umet_p [m/s]"""
        data = self._get_var(["u", "umet_p", "U"])
        if data is None:
            raise ValueError("Missing source variable for U Wind")
        return DataBuilder.build(
            data, self.ds, "umet_p", "U-component of Wind", "m s-1"
        )

    def calc_v_wind(self) -> xr.DataArray:
        """Output: vmet_p [m/s]"""
        data = self._get_var(["v", "vmet_p", "V"])
        if data is None:
            raise ValueError("Missing source variable for V Wind")
        return DataBuilder.build(
            data, self.ds, "vmet_p", "V-component of Wind", "m s-1"
        )

    def calc_mixing_ratio(self) -> xr.DataArray:
        """Output: QVAPOR_p [kg/kg]"""
        # Strategy 1: Specific Humidity (q)
        q = self._get_var(["q", "sh"])
        if q is not None:
            data = Thermodynamics.specific_humidity_to_mixing_ratio(q)
        else:
            # Strategy 2: Relative Humidity (r) + Temperature (t) + Pressure
            rh = self._get_var(["r", "rh"])
            t = self._get_var(["t", "tk_p", "T"])
            if rh is not None and t is not None:
                p_pa = self._get_pressure_pa()
                data = Thermodynamics.relative_humidity_to_mixing_ratio(rh, t, p_pa)
            else:
                # Strategy 3: Direct Mixing Ratio (QVAPOR)
                data = self._get_var(["QVAPOR_p", "QVAPOR"])
                if data is None:
                    raise ValueError(
                        "Cannot derive Mixing Ratio. Missing q, or (rh, t)."
                    )

        return DataBuilder.build(
            data, self.ds, "QVAPOR_p", "Water Vapor Mixing Ratio", "kg kg-1"
        )

    def calc_vertical_velocity(self) -> xr.DataArray:
        """Output: wa_p [m/s]"""
        # Strategy 1: Convert Omega (w/z/dp/dt) to w (m/s)
        omega = self._get_var(["w", "omega", "wap"])  # Pa/s
        if omega is not None:
            t = self._get_var(["t", "tk_p", "T"])
            p_pa = self._get_pressure_pa()

            # Try to get QVAPOR for virtual temperature correction
            try:
                q_da = self.calc_mixing_ratio()
                q = q_da.values
            except ValueError:
                q = None  # Fallback to dry T

            data = Thermodynamics.omega_to_w(omega, t, p_pa, q)
        else:
            # Strategy 2: Direct w (m/s)
            data = self._get_var(["wa_p", "W"])
            if data is None:
                # Fill with zeros if missing (common in some datasets)
                # But ideally should raise error if strictly required.
                # Here we assume standard fallback for coupling if vertical velocity is missing.
                t_ref = self._get_var(["t", "tk_p", "T"])
                data = np.zeros_like(t_ref)

        return DataBuilder.build(data, self.ds, "wa_p", "Vertical Velocity", "m s-1")

    # ===== Hydrometeors =====

    def _calc_hydro(
        self, target_name: str, era_name: str, wrf_name: str
    ) -> xr.DataArray:
        """Generic handler for hydrometeors."""
        # Strategy 1: ERA5 specific water content (kg/kg) -> Mixing Ratio
        val_era = self._get_var([era_name])
        if val_era is not None:
            data = Thermodynamics.specific_humidity_to_mixing_ratio(val_era)
        else:
            # Strategy 2: WRF/Direct Mixing Ratio
            val_wrf = self._get_var([wrf_name])
            if val_wrf is not None:
                data = val_wrf
            else:
                # Strategy 3: Missing (e.g. SFNO) -> Zeros
                # Find a template variable for shape
                t_ref = self._get_var(["t", "tk_p", "T"])
                data = np.zeros_like(t_ref)

        desc_map = {
            "QRAIN_p": "Rain Water Mixing Ratio",
            "QCLOUD_p": "Cloud Water Mixing Ratio",
            "QSNOW_p": "Snow Water Mixing Ratio",
            "QICE_p": "Cloud Ice Mixing Ratio",
            "QGRAUP_p": "Graupel Water Mixing Ratio",
        }
        return DataBuilder.build(
            data,
            self.ds,
            target_name,
            desc_map.get(target_name, target_name),
            "kg kg-1",
        )

    def calc_cloud_water(self):
        return self._calc_hydro("QCLOUD_p", "clwc", "QCLOUD_p")

    def calc_rain_water(self):
        return self._calc_hydro("QRAIN_p", "crwc", "QRAIN_p")

    def calc_ice_mixing(self):
        return self._calc_hydro("QICE_p", "ciwc", "QICE_p")

    def calc_snow_mixing(self):
        return self._calc_hydro("QSNOW_p", "cswc", "QSNOW_p")

    def calc_graupel_mixing(self):
        # ERA5/SFNO usually don't have Graupel
        return self._calc_hydro("QGRAUP_p", "graupel_missing", "QGRAUP_p")

    def calc_total_hydro(self) -> xr.DataArray:
        """Output: QTOTAL_p"""
        # Sum components
        qc = self.calc_cloud_water().values
        qr = self.calc_rain_water().values
        qi = self.calc_ice_mixing().values
        qs = self.calc_snow_mixing().values
        qg = self.calc_graupel_mixing().values

        total = qc + qr + qi + qs + qg
        return DataBuilder.build(
            total, self.ds, "QTOTAL_p", "Total Hydrometeors Mixing Ratio", "kg kg-1"
        )

    # ===== Surface Variables =====

    def calc_surface_temp(self) -> xr.DataArray:
        """Output: T2 [K]"""
        data = self._get_var(["2t", "t2m", "T2"])
        if data is None:
            raise ValueError("Missing source for 2m Temperature")
        return DataBuilder.build(
            data, self.ds, "T2", "2m Temperature", "K", is_3d=False
        )

    def calc_surface_mixing_ratio(self) -> xr.DataArray:
        """Output: Q2 [kg/kg]"""
        # Strategy 1: Dewpoint (2d) + Surface Pressure (sp)
        d2 = self._get_var(["2d", "d2m"])
        sp = self._get_var(["sp", "surface_pressure", "PSFC"])

        if d2 is not None and sp is not None:
            data = Thermodynamics.dewpoint_to_mixing_ratio(d2, sp)
        else:
            # Strategy 2: Direct Q2
            data = self._get_var(["Q2"])
            if data is None:
                raise ValueError("Missing source for Q2 (need 2d+sp or Q2)")

        return DataBuilder.build(
            data, self.ds, "Q2", "2m Mixing Ratio", "kg kg-1", is_3d=False
        )

    def calc_surface_rh(self) -> xr.DataArray:
        """Output: rh2 [%]"""
        # Strategy 1: T2 and D2
        t2 = self._get_var(["2t", "t2m", "T2"])
        d2 = self._get_var(["2d", "d2m"])

        if t2 is not None and d2 is not None:
            e = Thermodynamics.sat_vapor_pressure_water(d2)
            es = Thermodynamics.sat_vapor_pressure_water(t2)
            data = (e / es) * 100.0
            # Clip to 0-100
            data = np.clip(data, 0, 100)
        else:
            data = self._get_var(["rh2", "RH2"])
            if data is None:
                raise ValueError("Missing source for rh2")

        return DataBuilder.build(
            data, self.ds, "rh2", "2m Relative Humidity", "%", is_3d=False
        )


# ------------------------------------------------------------------------------
# Example Usage (Orchestrator)
# ------------------------------------------------------------------------------


def run_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """
    Entry point for the workflow.
    Generates all standard diagnostic variables required for coupling.
    """
    engine = DiagnosticEngine(ds)

    # List of diagnostics to run
    # This ensures One-Way (Era->WRF) and Two-Way (WRF->Era) compatibility
    # by forcing a standard output schema.
    results = []

    # Pressure Level Diagnostics
    try:
        results.append(engine.calc_geopotential_height())
    except ValueError as e:
        logger.warning(f"Could not calculate geopotential height: {e}")

    try:
        results.append(engine.calc_temperature())
    except ValueError as e:
        logger.warning(f"Could not calculate temperature: {e}")

    try:
        results.append(engine.calc_u_wind())
    except ValueError as e:
        logger.warning(f"Could not calculate u wind: {e}")

    try:
        results.append(engine.calc_v_wind())
    except ValueError as e:
        logger.warning(f"Could not calculate v wind: {e}")

    try:
        results.append(engine.calc_mixing_ratio())
    except ValueError as e:
        logger.warning(f"Could not calculate mixing ratio: {e}")

    try:
        results.append(engine.calc_vertical_velocity())
    except ValueError as e:
        logger.warning(f"Could not calculate vertical velocity: {e}")

    # Hydrometeors
    results.append(engine.calc_cloud_water())
    results.append(engine.calc_rain_water())
    results.append(engine.calc_ice_mixing())
    results.append(engine.calc_snow_mixing())
    results.append(engine.calc_graupel_mixing())
    results.append(engine.calc_total_hydro())

    # Surface Variables
    try:
        results.append(engine.calc_surface_temp())
    except ValueError as e:
        logger.warning(f"Could not calculate surface temperature: {e}")

    try:
        results.append(engine.calc_surface_mixing_ratio())
    except ValueError as e:
        logger.warning(f"Could not calculate surface mixing ratio: {e}")

    try:
        results.append(engine.calc_surface_rh())
    except ValueError as e:
        logger.warning(f"Could not calculate surface relative humidity: {e}")

    # Filter out None results if any diagnostic failed
    results = [res for res in results if res is not None]

    if not results:
        logger.warning("No diagnostic variables could be calculated.")
        return xr.Dataset()  # Return empty dataset if nothing was calculated

    return xr.merge(results)
