import importlib
import yaml

def load_diagnostics(yaml_path, context=None):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    reg_cfg = cfg.get("registry", {})
    diagnostics = {}

    for name, item in reg_cfg.items():
        func_name = item["function"]
        module = importlib.import_module("src.registry.diagnostic_functions")
        raw_func = getattr(module, func_name)

        def wrapped(ds, context=context, func=raw_func):
            return func(ds, context=context)

        diagnostics[name] = {
            "requires": item["requires"],
            "function": wrapped,
            "long_name": item.get("long_name", ""),
            "units": item.get("units", "")
        }

    return diagnostics

def sort_diagnostics_by_dependencies(diagnostics):
    sorted_list = []
    visited = set()

    def visit(var):
        if var in visited:
            return
        for dep in diagnostics[var]["requires"]:
            if dep in diagnostics:
                visit(dep)
        sorted_list.append(var)
        visited.add(var)

    for var in diagnostics:
        visit(var)

    return sorted_list

