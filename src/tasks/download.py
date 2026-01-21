from datetime import datetime, timedelta
import os
import cdsapi
from src.core.pipeline import Task
from src.core.context import Context


class BaseDownloader(Task):
    """Abstract base class for all downloaders."""

    pass


class ERA5Downloader(BaseDownloader):
    def execute(self, context: Context):
        print("[INFO] Starting ERA5 Downloader...")
        cfg = context.config

        # Determine timeline from config
        time_cfg = cfg["share"]["time_control"]
        start_t = datetime.strptime(time_cfg["start"], time_cfg["format"])
        end_t = datetime.strptime(time_cfg["end"], time_cfg["format"])
        step = timedelta(hours=time_cfg["base_step_hours"])

        total_steps = int((end_t - start_t) / step) + 1
        timeline = [start_t + step * i for i in range(total_steps)]

        client = cdsapi.Client()

        # Directories
        grib_dir = cfg["share"]["io_control"]["grib_dir"]
        os.makedirs(grib_dir, exist_ok=True)

        prefix = cfg["share"]["io_control"]["prefix"]
        timestr_fmt = prefix["timestr_fmt"]

        # We will loop here. In a more advanced Pipeline, the loop could be external.
        # But for now, we encapsulate the loop logic here as per original design flavor.

        for curr_time in timeline:
            timestamp = curr_time.strftime(timestr_fmt)
            combined_file = os.path.join(
                grib_dir, f"{prefix['combined']}_{timestamp}.grib"
            )

            # --- 1. Download Surface (SFC) first ---
            sl_cfg = cfg["download"]["dataset_surface"]
            sl_file = os.path.join(grib_dir, f"{prefix['surface']}_{timestamp}.grib")
            if not os.path.exists(sl_file):
                print(f"[DOWNLOAD] Fetching SFC: {timestamp}")
                req_sl = {
                    "product_type": "reanalysis",
                    "year": [curr_time.strftime("%Y")],
                    "month": [curr_time.strftime("%m")],
                    "day": [curr_time.strftime("%d")],
                    "time": [curr_time.strftime("%H:%M")],
                    "variable": sl_cfg["variables"],
                    "format": "grib",
                }
                client.retrieve(sl_cfg["title"], req_sl).download(sl_file)

            # --- 2. Download Pressure Level (PL) second ---
            pl_cfg = cfg["download"]["dataset_upper"]
            pl_file = os.path.join(grib_dir, f"{prefix['upper']}_{timestamp}.grib")
            if not os.path.exists(pl_file):
                print(f"[DOWNLOAD] Fetching PL: {timestamp}")
                req_pl = {
                    "product_type": "reanalysis",
                    "year": [curr_time.strftime("%Y")],
                    "month": [curr_time.strftime("%m")],
                    "day": [curr_time.strftime("%d")],
                    "time": [curr_time.strftime("%H:%M")],
                    "variable": pl_cfg["variables"],
                    "pressure_level": [str(level) for level in pl_cfg["levels"]],
                    "format": "grib",
                }
                client.retrieve(pl_cfg["title"], req_pl).download(pl_file)

            # --- 3. Merge into a single GRIB file ---
            if (
                os.path.exists(sl_file)
                and os.path.exists(pl_file)
                and not os.path.exists(combined_file)
            ):
                print(f"[MERGE] Concatenating GRIBs: {timestamp}")
                # Simple concatenation works for GRIB files
                with open(combined_file, "wb") as f_out:
                    with open(sl_file, "rb") as f_in:
                        f_out.write(f_in.read())
                    with open(pl_file, "rb") as f_in:
                        f_out.write(f_in.read())
                print(f"[DONE] Created merged GRIB: {combined_file}")


class GFSDownloader(BaseDownloader):
    """Reserved for GFS GRIB2 download."""

    def execute(self, context: Context):
        print("[INFO] GFS Download Reserved.")


class HimawariDownloader(BaseDownloader):
    """Reserved for HSD/Bz2 download."""

    def execute(self, context: Context):
        print("[INFO] Himawari Download Reserved.")
