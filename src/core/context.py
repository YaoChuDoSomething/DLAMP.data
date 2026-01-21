from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

import yaml


@dataclass
class Context:
    """
    Holds the shared state for the pipeline execution.
    """

    config_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    current_time: Optional[datetime] = None

    # Paths passing between tasks
    temp_grib_path: Optional[str] = None
    raw_nc_path: Optional[str] = None
    regrid_nc_path: Optional[str] = None
    final_nc_path: Optional[str] = None

    def load_config(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_config(self, key: str, default: Any = None) -> Any:
        # Simple dot notation support could be added here if needed
        return self.config.get(key, default)
