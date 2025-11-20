#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFNO Preprocessing Workflow
============================

Integrated preprocessing pipeline for SFNO global weather forecast model:
1. Run SFNO forecast
2. Convert output to ERA5-like format
3. Regrid to regional domain
4. Calculate diagnostic variables
5. Output in RWRF-compatible format

This script mirrors the DLAMPreproc.py workflow but uses SFNO forecasts
instead of ERA5 reanalysis data.

Author: Generated for DLAMP Project
Date: 2024
"""

import os
import sys
import argparse
from datetime import datetime
from loguru import logger

# Import preprocessing modules
from src.preproc.sfno_processor import SFNODataProcessor, SFNOFeedbackInterface
from src.preproc.dlamp_regridder import DataRegridder


###===== Workflow Control =========================================###
#
#  Main workflow switches
#
###================================================================###

# Workflow stages
DO_SFNO_FORECAST = True          # Run SFNO global forecast
DO_FORMAT_CONVERSION = True       # Convert SFNO output to ERA5-like format
DO_REGRIDDING = True             # Regrid to regional domain
DO_DIAGNOSTICS = True            # Calculate diagnostic variables

# Configuration
DLAMP_DATA_DIR = "./"
YAML_CONFIG = f"{DLAMP_DATA_DIR}/config/sfno.yaml"


###===== Helper Functions =========================================###

def setup_logger(log_dir="./logs"):
    """Configure loguru logger with file and console output"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sfno_preproc_{timestamp}.log")
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file logger
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="10 days"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def validate_config(yaml_path):
    """Validate configuration file exists and is readable"""
    if not os.path.exists(yaml_path):
        logger.error(f"Configuration file not found: {yaml_path}")
        raise FileNotFoundError(f"Config file missing: {yaml_path}")
    
    logger.info(f"Configuration file validated: {yaml_path}")


###===== SFNO Forecast Stage ======================================###

def run_sfno_forecast(yaml_config):
    """
    Stage 1: Run SFNO global weather forecast model
    
    This stage:
    - Loads SFNO model
    - Fetches initial conditions from GFS or CDS
    - Runs forecast for specified duration
    - Saves raw SFNO output
    
    Parameters
    ----------
    yaml_config : str
        Path to configuration file
        
    Returns
    -------
    list
        Timeline of forecast valid times
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: Running SFNO Global Forecast")
    logger.info("=" * 70)
    
    processor = SFNODataProcessor(yaml_config)
    
    # Run forecast and get timeline
    timeline = processor.run_forecast()
    
    logger.success(f"SFNO forecast completed. Generated {len(timeline)} timesteps.")
    return timeline, processor


def convert_sfno_format(processor, timeline):
    """
    Stage 2: Convert SFNO output to ERA5-like format
    
    This stage converts SFNO native output to ERA5 variable names and structure,
    enabling seamless integration with existing preprocessing pipeline.
    
    Parameters
    ----------
    processor : SFNODataProcessor
        Initialized SFNO processor
    timeline : list
        List of forecast valid times
        
    Returns
    -------
    list
        List of (time, pressure_level_file, surface_file) tuples
    """
    logger.info("=" * 70)
    logger.info("STAGE 2: Converting SFNO Output to ERA5 Format")
    logger.info("=" * 70)
    
    converted_files = []
    
    for curr_time in timeline:
        pl_nc, sl_nc = processor.convert_to_era5_format(curr_time)
        
        if pl_nc and sl_nc:
            converted_files.append((curr_time, pl_nc, sl_nc))
            logger.debug(f"Converted: {curr_time}")
    
    logger.success(f"Format conversion completed. {len(converted_files)} timesteps ready.")
    return converted_files


###===== Regridding and Diagnostics Stage =========================###

def run_regridding_and_diagnostics(yaml_config, converted_files):
    """
    Stage 3 & 4: Regrid to regional domain and calculate diagnostics
    
    This stage:
    - Horizontally interpolates from global grid to regional domain
    - Calculates derived meteorological variables
    - Outputs in RWRF-compatible format
    
    Parameters
    ----------
    yaml_config : str
        Path to configuration file
    converted_files : list
        List of (time, pl_file, sl_file) tuples from conversion stage
    """
    logger.info("=" * 70)
    logger.info("STAGE 3 & 4: Regridding and Diagnostic Calculation")
    logger.info("=" * 70)
    
    # Initialize regridder with SFNO configuration
    # The regridder now reads from sfnopl_* and sfnosl_* files
    regridder = DataRegridder(yaml_config)
    
    # Process each timestep
    timeline = regridder.build_timeline()
    
    logger.info(f"Processing {len(timeline)} timesteps for regional domain...")
    
    for curr_time in timeline:
        logger.info(f"Processing: {curr_time}")
        regridder.process_single_time(curr_time)
    
    logger.success("Regridding and diagnostic calculation completed.")


###===== Two-Way Feedback (Future) ================================###

def initialize_feedback_interface(yaml_config):
    """
    Initialize two-way feedback interface (future capability)
    
    This enables:
    - Regional model output to update SFNO global forecast
    - Boundary condition blending
    - Nested domain coupling
    
    Parameters
    ----------
    yaml_config : str
        Path to configuration file
        
    Returns
    -------
    SFNOFeedbackInterface
        Feedback interface object
    """
    logger.info("Initializing two-way feedback interface...")
    feedback = SFNOFeedbackInterface(yaml_config)
    
    if feedback.enabled:
        logger.info("Two-way feedback is ENABLED")
    else:
        logger.info("Two-way feedback is DISABLED (future capability)")
    
    return feedback


###===== Main Workflow ============================================###

def main():
    """
    Main preprocessing workflow orchestration
    """
    parser = argparse.ArgumentParser(
        description="SFNO Preprocessing Workflow for Regional Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow
  python SFNOPreproc.py
  
  # Use custom configuration
  python SFNOPreproc.py --config config/my_sfno.yaml
  
  # Skip forecast stage (use existing SFNO output)
  python SFNOPreproc.py --skip-forecast
  
  # Run only specific stages
  python SFNOPreproc.py --stages forecast,convert
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=YAML_CONFIG,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--skip-forecast',
        action='store_true',
        help='Skip SFNO forecast stage (use existing output)'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        default='all',
        help='Comma-separated list of stages to run: forecast,convert,regrid,diagnostics'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(args.log_dir)
    
    logger.info("=" * 70)
    logger.info("SFNO PREPROCESSING WORKFLOW")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Stages: {args.stages}")
    
    # Validate configuration
    validate_config(args.config)
    
    # Parse stages
    if args.stages == 'all':
        stages = ['forecast', 'convert', 'regrid', 'diagnostics']
    else:
        stages = [s.strip() for s in args.stages.split(',')]
    
    # Initialize feedback interface
    feedback = initialize_feedback_interface(args.config)
    
    try:
        # Stage 1: SFNO Forecast
        if 'forecast' in stages and not args.skip_forecast:
            timeline, processor = run_sfno_forecast(args.config)
        else:
            logger.info("Skipping SFNO forecast stage")
            processor = SFNODataProcessor(args.config)
            timeline = processor.create_timeline()
        
        # Stage 2: Format Conversion
        if 'convert' in stages:
            converted_files = convert_sfno_format(processor, timeline)
        else:
            logger.info("Skipping format conversion stage")
            converted_files = None
        
        # Stage 3 & 4: Regridding and Diagnostics
        if 'regrid' in stages or 'diagnostics' in stages:
            run_regridding_and_diagnostics(args.config, converted_files)
        else:
            logger.info("Skipping regridding and diagnostics stages")
        
        logger.success("=" * 70)
        logger.success("SFNO PREPROCESSING WORKFLOW COMPLETED SUCCESSFULLY")
        logger.success("=" * 70)
        
    except Exception as e:
        logger.exception(f"Workflow failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()