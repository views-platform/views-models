import math
from config_sweep import get_sweep_config

def calculate_combinations():
    """
    Calculates and prints the number of hyperparameter combinations in the sweep configuration.
    """
    sweep_config = get_sweep_config()
    parameters = sweep_config.get('parameters', {})
    method = sweep_config.get('method')

    total_combinations = 1
    discrete_params = {}
    continuous_params = []

    for name, config in parameters.items():
        if 'values' in config:
            num_values = len(config['values'])
            discrete_params[name] = num_values
            if num_values > 0:
                total_combinations *= num_values
        elif 'distribution' in config:
            continuous_params.append(name)

    print("Sweep Configuration Analysis")
    print("=" * 30)
    print(f"Sweep Method: '{method}'")
    print("\n--- Discrete Parameters ---")
    if not discrete_params:
        print("No discrete parameters with 'values' found.")
    else:
        for name, num in discrete_params.items():
            print(f"- {name}: {num} options")

    print("\n--- Continuous Parameters (Sampled) ---")
    if not continuous_params:
        print("No continuous parameters with 'distribution' found.")
    else:
        for name in continuous_params:
            print(f"- {name}")

    print("\n--- Summary ---")
    if method == 'grid':
        print(f"Total Combinations for a Grid Search: {total_combinations:,}")
    else:
        print(f"The total number of unique combinations from all discrete parameters is {total_combinations:,}.")
        print(f"\nHowever, since the sweep method is '{method}', it will NOT run every possible combination.")
        print("Instead, it will intelligently sample from the defined search space, including the continuous parameters.")
        print("The actual number of runs is controlled by your WandB sweep agent configuration (e.g., 'wandb agent --count <number_of_runs>').")


if __name__ == "__main__":
    calculate_combinations()
