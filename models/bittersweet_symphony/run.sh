if [[ "$OSTYPE" == "darwin"* ]]; then
  chsh -s /bin/zsh
  echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc                           
  echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
  echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
  source ~/.zshrc
fi

path="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" >/dev/null 2>&1 && pwd )"
echo $path
env_path="$path/envs/views_stepshifter"
eval "$(conda shell.bash hook)"

if [ -d "$env_path" ]; then
  echo "Conda environment already exists at $env_path. Checking dependencies..."
  conda activate $env_path
  echo "$env_path is activated"

  missing_packages=$(pip install --dry-run -r requirements.txt 2>&1 | grep "Requirement already satisfied" | wc -l)
  if [ "$missing_packages" -gt 0 ]; then
    echo "Installing missing or outdated packages..."
    pip install -r requirements.txt
  else
    echo "All packages are up-to-date."
  fi
else
  echo "Creating new Conda environment at $env_path..."
  conda create --prefix "$env_path" python=3.11 -y
  source activate $env_path
  pip install -r requirements.txt
fi

python main.py "$@"