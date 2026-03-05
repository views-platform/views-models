import importlib.util
import os
import sys

def load_module(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

models = ["novel_heuristics", "emerging_principles", "preliminary_directives"]

for model in models:
    print(f"\n--- Checking Model: {model} ---")
    hp_path = f"models/{model}/configs/config_hyperparameters.py"
    sweep_path = f"models/{model}/configs/config_sweep.py"
    
    if not os.path.exists(hp_path) or not os.path.exists(sweep_path):
        print(f"Skipping {model}: files not found.")
        continue
        
    hp_module = load_module(hp_path)
    sweep_module = load_module(sweep_path)
    
    hp_config = hp_module.get_hp_config()
    sweep_config_full = sweep_module.get_sweep_config()
    sweep_params = sweep_config_full.get('parameters', {})
    
    hp_keys = set(hp_config.keys())
    sweep_keys = set(sweep_params.keys())
    
    # Check for use_static_covariates specifically
    if "use_static_covariates" not in hp_keys:
        print("  MISSING: 'use_static_covariates' in config_hyperparameters")
    elif hp_config["use_static_covariates"] is not True:
        print(f"  WRONG VALUE: 'use_static_covariates' is {hp_config['use_static_covariates']} in config_hyperparameters")
        
    if "use_static_covariates" not in sweep_keys:
        print("  MISSING: 'use_static_covariates' in config_sweep")
    else:
        sweep_val = sweep_params["use_static_covariates"].get("values", [])
        if sweep_val != [True]:
            print(f"  WRONG VALUE: 'use_static_covariates' is {sweep_val} in config_sweep")

    only_hp = hp_keys - sweep_keys
    only_sweep = sweep_keys - hp_keys
    
    if only_hp:
        print(f"  Keys only in config_hyperparameters: {only_hp}")
    if only_sweep:
        print(f"  Keys only in config_sweep: {only_sweep}")
        
    common_keys = hp_keys & sweep_keys
    mismatches = []
    
    for key in common_keys:
        hp_val = hp_config[key]
        sweep_val_struct = sweep_params[key]
        
        # Sweep values are usually in a list under 'values'
        if 'values' in sweep_val_struct:
            sweep_vals = sweep_val_struct['values']
            if len(sweep_vals) == 1:
                # If there's only one value in sweep, it should match hp_val
                if sweep_vals[0] != hp_val:
                    mismatches.append((key, hp_val, sweep_vals[0]))
            else:
                # If multiple values, check if hp_val is one of them
                if hp_val not in sweep_vals:
                    mismatches.append((key, hp_val, f"NOT IN {sweep_vals}"))
        else:
            # Could be other sweep types (distribution etc), but here it seems to be mostly grid/list
            print(f"  Key '{key}' has non-standard sweep structure: {sweep_val_struct}")

    if mismatches:
        print("  Value mismatches (Key, HP Value, Sweep Value[0] or List):")
        for m in mismatches:
            print(f"    - {m[0]}: HP={m[1]}, Sweep={m[2]}")
    else:
        if not only_hp and not only_sweep:
            print("  All keys and single-value parameters match exactly.")
        else:
            print("  Common keys match in values.")

