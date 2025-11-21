import yaml
import importlib
from typing import Dict, Any, List
from datetime import datetime
import os
import torch

from .utils.file_utils import get_logger

logger = get_logger("WorkflowEngine")

class WorkflowConfigurationError(Exception):
    """Configuration parsing errors."""
    pass

class WorkflowExecutionError(Exception):
    """Runtime execution errors."""
    pass

class WorkflowEngine:
    """
    Generic Workflow Executor.
    Strictly follows SRP: It only executes steps defined in the configuration.
    It knows nothing about 'weather', 'models', or 'coupling'.
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.workflow_state: Dict[str, Any] = {} # Stores objects and data produced by steps

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise WorkflowConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise WorkflowConfigurationError(f"Error parsing YAML configuration: {e}")

    def _resolve_placeholders(self, value: Any, context: Dict[str, Any]) -> Any:
        """Recursively resolves placeholders in strings, dictionaries, and lists."""
        if isinstance(value, str):
            if value.startswith('{') and value.endswith('}'):
                # Handle direct object references like "{some_object}"
                key = value[1:-1]
                if key in context:
                    return context[key]
                # Try to resolve nested dictionary references e.g., "{resources.file_path}"
                parts = key.split('.')
                resolved_val = context
                try:
                    for part in parts:
                        resolved_val = resolved_val[part]
                    return resolved_val
                except (KeyError, TypeError):
                    pass # Not a resolvable object reference, treat as format string
            
            # Fallback to string formatting
            try:
                # Provide common_settings, resources, and current workflow_state for formatting
                format_context = {
                    **self.config.get('common_settings', {}),
                    **self.config.get('resources', {}),
                    **context, # This includes loop_item and step_index
                    **self.workflow_state # Allow referencing results from previous steps
                }
                # logger.debug(f"Attempting to format: {value} with context keys: {format_context.keys()}")
                return value.format(**format_context)
            except KeyError as e:
                logger.warning(f"Failed to resolve placeholder '{e}' in string '{value}'. It will be kept as is.")
                return value
            except AttributeError: # Happens if value is not a string but has {}
                return value
        elif isinstance(value, dict):
            return {self._resolve_placeholders(k, context): self._resolve_placeholders(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_placeholders(item, context) for item in value]
        return value

    def run_workflow(self, workflow_name: str, dynamic_context: Optional[Dict[str, Any]] = None):
        """Executes a specified workflow from the loaded configuration."""
        if workflow_name not in self.config.get('workflows', {}):
            raise WorkflowExecutionError(f"Workflow '{workflow_name}' not found in configuration.")

        logger.info(f"Starting workflow: {workflow_name}")
        workflow_definition = self.config['workflows'][workflow_name]
        steps = workflow_definition.get('steps', [])
        
        # Initialize workflow_state for this run, including dynamic_context
        self.workflow_state = {
            "common_settings": self.config.get('common_settings', {}),
            "resources": self.config.get('resources', {}),
        }
        if dynamic_context:
            self.workflow_state.update(dynamic_context)

        for step_index, step_config in enumerate(steps):
            step_name = step_config.get('name', f"Unnamed Step {step_index}")
            logger.info(f"Executing step {step_index + 1}: {step_name}")
            
            current_step_context = {
                "step_index": step_index,
                "step_name": step_name,
                **self.workflow_state # Current state is part of context for resolving
            }

            if step_config.get('type') == 'iterator':
                self._execute_iterator_step(step_config, current_step_context)
            else:
                self._execute_single_step(step_config, current_step_context)

        logger.info(f"Workflow '{workflow_name}' completed successfully.")
        return self.workflow_state # Return final state for potential inspection

    def _execute_single_step(self, step_config: Dict[str, Any], context: Dict[str, Any]):
        """Executes a single non-iterator step."""
        step_name = step_config.get('name', 'Unnamed Step')
        module_path = step_config.get('module')
        class_name = step_config.get('class')
        action_name = step_config.get('action')
        output_key = step_config.get('output_key')
        instance_key = step_config.get('instance_key')
        ignore_errors = step_config.get('ignore_errors', False)
        condition = step_config.get('condition')

        try:
            # Check condition if present
            if condition:
                # Evaluate condition string as a Python expression
                # Make workflow_state available for evaluation
                evaluated_condition = eval(condition, {}, self.workflow_state)
                if not evaluated_condition:
                    logger.info(f"Skipping step '{step_name}' due to unmet condition: '{condition}'")
                    return

            module = importlib.import_module(module_path)
            
            target_obj = None
            if instance_key and instance_key in self.workflow_state:
                target_obj = self.workflow_state[instance_key]
            elif class_name:
                target_class = getattr(module, class_name)
                # Filter args for instantiation
                init_args_config = step_config.get('args', {})
                init_args = self._resolve_placeholders(init_args_config, context)
                
                if action_name == "instantiate": # Special action to just instantiate
                    target_obj = target_class(**init_args)
                else: # Instantiate and then call a method
                    target_obj = target_class(**init_args)
            elif action_name: # Standalone function
                target_obj = module # Module itself is the target
            else:
                raise WorkflowConfigurationError(f"Step '{step_name}' requires 'class' or 'action'.")

            result = None
            if action_name and action_name != "instantiate":
                method_to_call = getattr(target_obj, action_name)
                method_args_config = step_config.get('args', {})
                method_args = self._resolve_placeholders(method_args_config, context)
                result = method_to_call(**method_args)
            elif target_obj is not None and not class_name: # Case of a standalone function directly called
                 func_args_config = step_config.get('args', {})
                 func_args = self._resolve_placeholders(func_args_config, context)
                 result = getattr(module, action_name)(**func_args) # Assuming action_name is the function name

            if output_key:
                self.workflow_state[output_key] = result
                logger.debug(f"Step '{step_name}' stored result in '{output_key}'.")

        except Exception as e:
            if ignore_errors:
                logger.warning(f"Step '{step_name}' failed but errors are ignored: {e}")
                if output_key:
                    self.workflow_state[output_key] = None # Ensure output key exists but is None
            else:
                raise WorkflowExecutionError(f"Error in step '{step_name}': {e}") from e

    def _execute_iterator_step(self, step_config: Dict[str, Any], context: Dict[str, Any]):
        """Handles iterator steps which loop through results of a generator."""
        step_name = step_config.get('name', 'Unnamed Iterator Step')
        source_object_key = step_config.get('source_object')
        method_name = step_config.get('method')
        method_args_config = step_config.get('method_args', {})
        loop_limit = step_config.get('loop_limit')
        loop_item_key = step_config.get('loop_item_key', 'loop_item')
        inner_steps = step_config.get('steps', [])

        if source_object_key not in self.workflow_state:
            raise WorkflowConfigurationError(f"Iterator step '{step_name}' source object '{source_object_key}' not found in state.")
        
        source_object = self.workflow_state[source_object_key]
        method_to_call = getattr(source_object, method_name)
        
        # Resolve args for the iterator creation method
        method_args = self._resolve_placeholders(method_args_config, context)
        
        iterator = method_to_call(**method_args)

        logger.info(f"Entering iterator loop for '{step_name}' with limit {loop_limit}.")
        for i in range(loop_limit):
            try:
                item = next(iterator)
                logger.info(f"Iterator step '{step_name}' - Iteration {i+1}/{loop_limit}")
                
                # Add loop-specific context
                loop_context = {
                    "current_iteration": i,
                    "step_index": i, # Often used as step_index in loop
                    loop_item_key: item,
                    **self.workflow_state # Current state
                }

                # Store the current item in workflow_state for inner steps to access
                self.workflow_state[loop_item_key] = item

                for inner_step_index, inner_step_config in enumerate(inner_steps):
                    inner_step_name = inner_step_config.get('name', f"Inner Step {inner_step_index}")
                    logger.debug(f"  Executing inner step: {inner_step_name} in iteration {i+1}")
                    
                    # Pass the combined context for resolving placeholders in inner steps
                    self._execute_single_step(inner_step_config, loop_context)
                    
            except StopIteration:
                logger.info(f"Iterator for '{step_name}' exhausted after {i} iterations.")
                break
            except Exception as e:
                raise WorkflowExecutionError(f"Error in iterator step '{step_name}' at iteration {i+1}: {e}") from e
        logger.info(f"Exiting iterator loop for '{step_name}'.")

# Example for standalone execution
if __name__ == "__main__":
    import argparse
    import sys
    
    # Create a dummy config file for testing
    dummy_config_path = "config/temp_workflow_config.yaml"
    os.makedirs("config", exist_ok=True) # Ensure config directory exists
    
    # Define a simple dummy config that uses some of the new modules
    dummy_config_content = """
common_settings:
  output_base_dir: "test_outputs"
  device: "cpu"

resources:
  initial_condition_file: "dummy_initial_data.nc"

workflows:
  test_workflow:
    steps:
      - name: "Initialize Model Loader"
        module: "src.op.global_operators"
        class: "ModelLoader"
        action: "instantiate"
        args:
          device: "{common_settings.device}"
        output_key: "loader"

      - name: "Create Dummy NC File"
        module: "__main__" # Refers to this script for dummy data creation
        action: "create_dummy_nc_file"
        args:
          path: "{resources.initial_condition_file}"
        output_key: "dummy_file_created"

      - name: "Load Initial Conditions"
        instance_key: "loader"
        action: "load_nc_file"
        args:
          path: "{resources.initial_condition_file}"
        output_key: "current_state"
        
      - name: "Initialize Exporter"
        module: "src.op.global_operators"
        class: "DataExporter"
        action: "instantiate"
        args:
          output_dir: "{common_settings.output_base_dir}/test_workflow"
        output_key: "exporter"
        
      - name: "Save Initial State"
        instance_key: "exporter"
        action: "save"
        args:
          ds: "{current_state}"
          prefix: "initial_state"
    """
    
    with open(dummy_config_path, 'w') as f:
        f.write(dummy_config_content)
    
    # Dummy function to create a NetCDF file for testing
    def create_dummy_nc_file(path: str):
        logger.info(f"Creating dummy NetCDF file at {path}")
        # Create a simple xarray Dataset
        import xarray as xr
        import numpy as np
        import pandas as pd
        
        times = pd.to_datetime(["2023-01-01T00:00:00"])
        lats = np.linspace(0, 10, 5)
        lons = np.linspace(0, 10, 5)
        data = np.random.rand(1, 5, 5)
        
        dummy_ds = xr.Dataset(
            {
                "t2m": (("time", "lat", "lon"), data * 273),
                "u10": (("time", "lat", "lon"), data * 10),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dummy_ds.to_netcdf(path)
        logger.info(f"Dummy file created: {path}")

    parser = argparse.ArgumentParser(description="Run a workflow using the WorkflowEngine.")
    parser.add_argument("--config", type=str, default=dummy_config_path,
                        help="Path to the workflow configuration YAML file.")
    parser.add_argument("--workflow", type=str, default="test_workflow",
                        help="Name of the workflow to run from the config file.")
    args = parser.parse_args()

    # Initialize these variables before the try block to ensure they are available in finally
    engine = None
    config_data = {}
    resources = {}
    common_settings = {}

    try:
        engine = WorkflowEngine(args.config)
        config_data = engine.config # Get the loaded config data
        resources = config_data.get('resources', {})
        common_settings = config_data.get('common_settings', {})

        final_state = engine.run_workflow(args.workflow)
        logger.info("Workflow execution finished.")
        # Optional: inspect final_state if needed
        # print("Final workflow state keys:", final_state.keys())

    except (WorkflowConfigurationError, WorkflowExecutionError) as e:
        logger.error(f"Workflow error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Clean up dummy file and directory
        # Use the resources and common_settings extracted from the loaded config
        initial_condition_file_path = resources.get('initial_condition_file')
        output_base_dir_path = common_settings.get('output_base_dir')

        if initial_condition_file_path and os.path.exists(initial_condition_file_path):
            os.remove(initial_condition_file_path)
            logger.info(f"Cleaned up dummy file: {initial_condition_file_path}")
        
        if output_base_dir_path and os.path.exists(output_base_dir_path):
            import shutil
            shutil.rmtree(output_base_dir_path)
            logger.info(f"Cleaned up output directory: {output_base_dir_path}")

        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
            logger.info(f"Cleaned up dummy config file: {dummy_config_path}")

