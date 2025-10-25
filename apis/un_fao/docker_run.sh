#!/bin/bash

script_path=$(dirname "$(realpath "$0")")
project_path="$( cd "$script_path/../../" >/dev/null 2>&1 && pwd )"
env_path="$project_path/envs/views-faoapi"

# In Docker, we don't use Conda, we use the system Python
echo "Running in Docker environment - using system Python"

# Check if we're in Docker (optional)
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
    
    # Install the private package if not already installed
    if ! python -c "import views_faoapi" &> /dev/null; then
        echo "Installing views-faoapi from private repository..."
        
        # Use GitHub token from environment if available
        if [ -n "$GITHUB_TOKEN" ]; then
            pip install git+https://${GITHUB_TOKEN}@github.com/views-platform/views-faoapi.git@development
        else
            echo "Error: GITHUB_TOKEN environment variable not set"
            echo "Please set GITHUB_TOKEN to install the private package"
            exit 1
        fi
    else
        echo "views-faoapi package already installed"
    fi
else
    # Local development with Conda (original logic)
    eval "$(conda shell.bash hook)"

    if [ -d "$env_path" ]; then
        echo "Conda environment already exists at $env_path. Checking dependencies..."
        conda activate "$env_path"
        echo "$env_path is activated"

        missing_packages=$(pip install --dry-run -r $script_path/requirements.txt 2>&1 | grep -v "Requirement already satisfied" | wc -l)
        if [ "$missing_packages" -gt 0 ]; then
            echo "Installing missing or outdated packages..."
            pip install -r $script_path/requirements.txt
        else
            echo "All packages are up-to-date."
        fi
    else
        echo "Creating new Conda environment at $env_path..."
        conda create --prefix "$env_path" python=3.11 -y
        conda activate "$env_path"
        pip install git+https://github.com/views-platform/views-faoapi.git@development
    fi
fi

echo "Running $script_path/main.py"
python $script_path/main.py "$@"