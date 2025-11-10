import argparse
import yaml
import sys
import os

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.workflows import data_processing, sfno_forecast

def run_data_ingestion(config_path):
    """Runs the Data Ingestion and RWRF Conversion Workflow."""
    print(f"Running data ingestion with config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_processing.main(config, config_path)

def run_sfno_forecast(config_path):
    """Runs the One-Way SFNO-GFS Forecast Workflow."""
    print(f"Running SFNO forecast with config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    sfno_forecast.main(config, config_path)

def main():
    parser = argparse.ArgumentParser(description="Earth2Studio Workflow Manager")
    parser.add_argument(
        "workflow",
        choices=["ingest", "forecast"],
        help="The workflow to execute: 'ingest' for data processing, 'forecast' for SFNO forecast."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the selected workflow."
    )

    args = parser.parse_args()

    if args.workflow == "ingest":
        run_data_ingestion(args.config)
    elif args.workflow == "forecast":
        run_sfno_forecast(args.config)

if __name__ == "__main__":
    main()