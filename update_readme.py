import importlib.util
import os
from pathlib import Path
import re
from views_pipeline_core.managers.model import ModelManager, ModelPathManager


base_dir = os.getcwd()
print(base_dir)
target_dir = Path(base_dir+"/views-models/models/")
target_dir

# Update repository structure:
def generate_repo_structure(folders, scripts, model_name, root_key="model_dir"):
    """Generate a structured repository tree with correct folder hierarchy and script placement."""

    if root_key not in folders:
        raise ValueError(f"Root key '{root_key}' not found in folders dictionary")

    root_path = Path(folders[root_key])  # Get the main model directory path
    tree = [model_name]  # Start with the model name
    folder_structure = {}

    # Build folder structure and ensure all folders exist in the mapping
    for folder_name, folder_path in folders.items():
        path = Path(folder_path)
        relative_path = path.relative_to(root_path)
        folder_structure[str(relative_path)] = {"name": folder_name, "scripts": []}

    # Assign scripts to the correct folders
    for script_name, script_path in scripts.items():
        script_path_obj = Path(script_path)
        parent_folder = script_path_obj.parent.relative_to(root_path)

        # Ensure the parent folder exists in our dictionary before adding
        parent_folder_str = str(parent_folder)
        if parent_folder_str in folder_structure:
            folder_structure[parent_folder_str]["scripts"].append(script_name)

    # Generate the tree output
    seen_folders = set()

    def build_tree(path, depth=0):
        """Recursive function to format the tree output."""
        indent = "│   " * depth
        path_str = str(path)

        # Ensure we don't print duplicate folders
        if path_str in seen_folders:
            return
        seen_folders.add(path_str)

        tree.append(f"{indent}├── {path.name}")

        # Add scripts belonging to this folder
        if path_str in folder_structure:
            for script in sorted(folder_structure[path_str]["scripts"]):
                tree.append(f"{indent}│   ├── {script}")

        # Recurse into subfolders
        subfolders = [p for p in folder_structure if Path(p).parent == path]
        for subfolder in sorted(subfolders):
            build_tree(Path(subfolder), depth + 1)

    # Build tree from the root folder
    build_tree(Path("."))  # "." represents the root of the model repo

    # Add root-level files (only if they are not assigned elsewhere)
    root_scripts = set(scripts.keys()) - {s for f in folder_structure.values() for s in f["scripts"]}
    root_files = ["requirements.txt", "run.sh"] + sorted(root_scripts)

    for file in root_files:
        tree.append(f"├── {file}")

    return "\n".join(tree)


for subfolder in target_dir.iterdir():
    if subfolder.is_dir():  # Check if it's a directory
        print(f"Model: {subfolder.name}")
        #configs_dir = Path(subfolder.name+"/configs")
        configs_dir = target_dir / subfolder.name / "configs"
        model_manager = ModelManager(model_path=ModelPathManager(configs_dir))
        mpm = ModelPathManager(configs_dir)

        ## Get Meta Info
        model_name = model_manager.configs['name']
        model_name = " ".join(word.capitalize() for word in model_name.split("_"))

        algorithm = model_manager.configs['algorithm']
        if algorithm=='HurdleModel':
            classifier = model_manager.configs['model_clf']
            regressor = model_manager.configs['model_reg']
            algorithm_all = f"{algorithm} (Classifier: {classifier}, Regressor: {regressor})"
        else:
            algorithm_all = algorithm

        target = model_manager.configs['depvar']
        if isinstance(target, list):
            target = ", ".join(target)
        queryset = model_manager.configs['queryset']
        level = model_manager.configs['level']
        try:
            metrics = model_manager.configs['metrics']
        except KeyError:
            metrics = "No information provided"
        if isinstance(metrics, list):
            metrics = ", ".join(metrics)

        ## Get deployment mode 
        deployment = model_manager.configs['deployment_status']

        ## Get queryset description
        queryset_info = mpm.get_queryset()
        description = queryset_info.description
        try:
            description = " ".join(description.split())
        except AttributeError:
            description = 'No description provided'
        name = queryset_info.name

        ## Update old README file - For Bitter Symphony Model 
        scaffold_path = target_dir / "README_scaffold.md"
        readme_path = target_dir / subfolder.name / "README.md"

        # Read old README
        with open(readme_path, "r") as file:
            old_readme_content = file.read()

        # Add created sessioin if it exists

        match = re.search(r"(## Created on.*)", old_readme_content, re.DOTALL)
        if match==None:
            new_string=''
        else:
            created_section = match.group(1).strip()
            insert_position = created_section.find("##")

            # Find where the '##' ends (after '##' and the next space)
            end_of_heading = len("##")  # Skip the '##' part itself
            new_string = created_section[:end_of_heading] + " " + 'Model' + created_section[end_of_heading:]

        # Read scaffold.md content
        with open(scaffold_path, "r") as file:
            content = file.read()


        # Dictionary of placeholders and their replacements
        replacements = {
            "{{MODEL_NAME}}": model_name,
            "{{MODEL_ALGORITHM}}": algorithm_all,
            "{{LEVEL_OF_ANALYSIS}}": level,
            "{{TARGET}}": target,
            "{{FEATURES}}": name, 
            "{{DESCRIPTION}}": description,
            "{{DEPLOYMENT}}": deployment,
            "{{METRICS}}": metrics,
            "{{CREATED_SECTION}}": new_string,
        }


        # Replace placeholders in scaffold content
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)


        repo_root = target_dir / subfolder.name
        repo_structure = generate_repo_structure(mpm.get_directories(), mpm.get_scripts(), model_name=model_name)
        formatted_structure = f"```\n{repo_structure}\n```"
        formatted_structure


        updated_readme = content.replace("## Repository Structure",
                f"## Repository Structure\n\n{formatted_structure}",
            )
        
        # Write the updated content to README.md
        with open(readme_path, "w") as file:
            file.write(updated_readme)

# Update repository structure:
def generate_tree(directory, prefix=""):
    """Recursively generate a text-based tree structure for a directory."""
    tree = []
    entries = sorted(os.listdir(directory))  # Sort to maintain consistent order
    entries = [e for e in entries if not e.startswith(".") and e != "__pycache__"]  # Ignore hidden files
    
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = (index == len(entries) - 1)  # Check if it's the last item
        
        # Tree structure formatting
        connector = "└── " if is_last else "├── "
        tree.append(f"{prefix}{connector}{entry}")
        
        if os.path.isdir(path):  # Recursively add subdirectories
            extension = "    " if is_last else "│   "
            tree.extend(generate_tree(path, prefix + extension))
    
    return tree


for subfolder in target_dir.iterdir():
    if subfolder.is_dir():  # Check if it's a directory
        print(f"Model: {subfolder.name}")
        #configs_dir = Path(subfolder.name+"/configs")
        configs_dir = target_dir / subfolder.name / "configs"
        config_modules = {}

        # Iterate through all .py files in the configs folder
        for config_file in configs_dir.glob("*.py"):
            module_name = config_file.stem  # Get filename without .py extension
            spec = importlib.util.spec_from_file_location(module_name, str(config_file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    
            # Store the module in a dictionary
            config_modules[module_name] = module

        ## Get Meta Info

        meta_info = config_modules["config_meta"].get_meta_config()

        algorithm = meta_info['algorithm']
        if algorithm=='HurdleModel':
            classifier = meta_info['model_clf']
            regressor = meta_info['model_reg']
            algorithm_all = f"{algorithm} (Classifier: {classifier}, Regressor: {regressor})"
        else:
            algorithm_all = algorithm

        target = meta_info['depvar']
        if isinstance(target, list):
            target = ", ".join(target)
        queryset = meta_info['queryset']
        level = meta_info['level']
        try:
            metrics = meta_info['metrics']
        except KeyError:
            metrics = "No information provided"
        if isinstance(metrics, list):
            metrics = ", ".join(metrics)
        model_name = meta_info['name']
        model_name = " ".join(word.capitalize() for word in model_name.split("_"))

        ## Get deployment mode 
        deployment_info = config_modules["config_deployment"].get_deployment_config()
        deployment = deployment_info['deployment_status']
        deployment
        ## Get queryset description

        queryset_info = config_modules["config_queryset"].generate()
        queryset_info
        description = queryset_info.description
        try:
            description = " ".join(description.split())
        except AttributeError:
            description = 'No description provided'
        name = queryset_info.name

        ## Update old README file - For Bitter Symphony Model 
        scaffold_path = target_dir / "README_scaffold.md"
        readme_path = target_dir / subfolder.name / "README.md"

        # Read old README
        with open(readme_path, "r") as file:
            old_readme_content = file.read()

        # Add created sessioin if it exists

        match = re.search(r"(## Created on.*)", old_readme_content, re.DOTALL)
        if match==None:
            new_string=''
        else:
            created_section = match.group(1).strip()
            insert_position = created_section.find("##")

            # Find where the '##' ends (after '##' and the next space)
            end_of_heading = len("##")  # Skip the '##' part itself
            new_string = created_section[:end_of_heading] + " " + 'Model' + created_section[end_of_heading:]

        # Read scaffold.md content
        with open(scaffold_path, "r") as file:
            content = file.read()


        # Dictionary of placeholders and their replacements
        replacements = {
            "{{MODEL_NAME}}": model_name,
            "{{MODEL_ALGORITHM}}": algorithm_all,
            "{{LEVEL_OF_ANALYSIS}}": level,
            "{{TARGET}}": target,
            "{{FEATURES}}": name, 
            "{{DESCRIPTION}}": description,
            "{{DEPLOYMENT}}": deployment,
            "{{METRICS}}": metrics,
            "{{CREATED_SECTION}}": new_string,
        }


        # Replace placeholders in scaffold content
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)


        #repo_root = "/home/sonja/Desktop/views-platform/views-models/models/bittersweet_symphony"
        repo_root = target_dir / subfolder.name
        repo_structure = "\n".join(generate_tree(repo_root))
        repo_structure
        formatted_structure = f"```\n{repo_structure}\n```"
        formatted_structure


        updated_readme = content.replace("## Repository Structure",
                f"## Repository Structure\n\n{formatted_structure}",
            )

        
        # Write the updated content to README.md
        with open(readme_path, "w") as file:
            file.write(updated_readme)

##############################################
####             Ensembles                ####
##############################################

base_dir = os.getcwd()
target_ensemble_dir = Path(base_dir+"/views-models/ensembles/")
#target_dir = Path(base_dir+"/views-models/models/")
target_ensemble_dir


for subfolder in target_ensemble_dir.iterdir():
    if subfolder.is_dir():  # Check if it's a directory
        print(f"Ensemble: {subfolder.name}")
        #configs_dir = Path(subfolder.name+"/configs")
        configs_dir = target_ensemble_dir / subfolder.name / "configs"
        #configs_dir = "/views-models/ensembles/" + subfolder.name + "/configs"
        config_modules = {}

        # Iterate through all .py files in the configs folder
        for config_file in configs_dir.glob("*.py"):
            module_name = config_file.stem  # Get filename without .py extension
            spec = importlib.util.spec_from_file_location(module_name, str(config_file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    
            # Store the module in a dictionary
            config_modules[module_name] = module

        ## Get Meta Info

        meta_info = config_modules["config_meta"].get_meta_config()

        name = meta_info['name']
        model_name = " ".join(word.capitalize() for word in name.split("_"))
        models = meta_info['models']
        if isinstance(models, list):
            models = ", ".join(models)
        try:
            metrics = meta_info['metrics']
        except KeyError:
            metrics = "No information provided"
        if isinstance(metrics, list):
            metrics = ", ".join(metrics)
        target = meta_info['depvar']
        if isinstance(target, list):
            target = ", ".join(target)

        aggregation = meta_info['aggregation']

        ## Get deployment mode 
        deployment_info = config_modules["config_deployment"].get_deployment_config()
        deployment = deployment_info['deployment_status']

        ## Update old README file - For Bitter Symphony Model 
        scaffold_path = target_dir / "README_scaffold.md"
        readme_path = target_dir / subfolder.name / "README.md"

        # Read old README
        with open(readme_path, "r") as file:
            old_readme_content = file.read()

        # Add created sessioin if it exists

        match = re.search(r"(## Created on.*)", old_readme_content, re.DOTALL)
        if match==None:
            new_string=''
        else:
            created_section = match.group(1).strip()
            insert_position = created_section.find("##")

            # Find where the '##' ends (after '##' and the next space)
            end_of_heading = len("##")  # Skip the '##' part itself
            new_string = created_section[:end_of_heading] + " " + 'Model' + created_section[end_of_heading:]

        # Read scaffold.md content
        with open(scaffold_path, "r") as file:
            content = file.read()


        # Dictionary of placeholders and their replacements
        replacements = {
            "{{MODEL_NAME}}": model_name,
            "{{MODEL_ALGORITHM}}": algorithm_all,
            "{{LEVEL_OF_ANALYSIS}}": level,
            "{{TARGET}}": target,
            "{{FEATURES}}": name, 
            "{{DESCRIPTION}}": description,
            "{{DEPLOYMENT}}": deployment,
            "{{METRICS}}": metrics,
            "{{CREATED_SECTION}}": new_string,
        }


        # Replace placeholders in scaffold content
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)


        #repo_root = "/home/sonja/Desktop/views-platform/views-models/models/bittersweet_symphony"
        repo_root = target_dir / subfolder.name
        repo_structure = "\n".join(generate_tree(repo_root))
        repo_structure
        formatted_structure = f"```\n{repo_structure}\n```"
        formatted_structure


        updated_readme = content.replace("## Repository Structure",
                f"## Repository Structure\n\n{formatted_structure}",
            )

        
        # Write the updated content to README.md
        with open(readme_path, "w") as file:
            file.write(updated_readme)

            

for i in range(0,5)[:1]:
    print(i)

configs_dir = Path("views-models/ensembles/cruel_summer/configs")
configs_dir
config_modules = {}

# Iterate through all .py files in the configs folder
for config_file in configs_dir.glob("*.py"):
    module_name = config_file.stem  # Get filename without .py extension
    spec = importlib.util.spec_from_file_location(module_name, str(config_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Store the module in a dictionary
    config_modules[module_name] = module

config_modules

## Get Meta Info

meta_info = config_modules["config_meta"].get_meta_config()

algorithm = meta_info['algorithm']
if algorithm=='HurdleModel':
    classifier = meta_info['model_clf']
    regressor = meta_info['model_reg']
    algorithm_all = f"{algorithm} (Classifier: {classifier}, Regressor: {regressor})"
else:
    algorithm_all = algorithm

algorithm_all
target = meta_info['depvar']
if isinstance(target, list):
        target = ", ".join(target)
queryset = meta_info['queryset']
level = meta_info['level']
metrics = meta_info['metrics']
model_name = meta_info['name']

model_name = " ".join(word.capitalize() for word in model_name.split("_"))

model_name

## Get deployment mode 
deployment_info = config_modules["config_deployment"].get_deployment_config()
deployment = deployment_info['deployment_status']
deployment
## Get queryset description

queryset_info = config_modules["config_queryset"].generate()
queryset_info
description = queryset_info.description
print(description)
try:
    description = " ".join(description.split())
except AttributeError:
    description = 'No description provided'
    
name = queryset_info.name


## Update old README file - For Bitter Symphony Model 
scaffold_path = Path("/home/sonja/Desktop/views-platform/views-models/README_scaffold.md")
readme_path = Path("/home/sonja/Desktop/views-platform/views-models/models/blank_space/README.md")


# Read old README
with open(readme_path, "r") as file:
    old_readme_content = file.read()


match = re.search(r"(## Created on.*)", old_readme_content, re.DOTALL)
if match == None:
    pass
else:
    created_section = match.group(1).strip()

    insert_position = created_section.find("##")


    # Find where the '##' ends (after '##' and the next space)
    end_of_heading = len("##")  # Skip the '##' part itself
    new_string = created_section[:end_of_heading] + " " + 'Model' + created_section[end_of_heading:]

# Read scaffold.md content
with open(scaffold_path, "r") as file:
    content = file.read()


# Dictionary of placeholders and their replacements
replacements = {
    "{{MODEL_NAME}}": model_name,
    "{{MODEL_ALGORITHM}}": algorithm_all,
    "{{LEVEL_OF_ANALYSIS}}": level,
    "{{TARGET}}": target,
    "{{FEATURES}}": name, 
    "{{DESCRIPTION}}": description,
    "{{DEPLOYMENT}}": deployment,
    "{{METRICS}}": ", ".join(metrics),
    "{{CREATED_SECTION}}": new_string,
}

replacements

# Replace placeholders in scaffold content
for placeholder, value in replacements.items():
    content = content.replace(placeholder, value)

content

# Update repository structure:
def generate_tree(directory, prefix=""):
    """Recursively generate a text-based tree structure for a directory."""
    tree = []
    entries = sorted(os.listdir(directory))  # Sort to maintain consistent order
    entries = [e for e in entries if not e.startswith(".")]  # Ignore hidden files
    
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = (index == len(entries) - 1)  # Check if it's the last item
        
        # Tree structure formatting
        connector = "└── " if is_last else "├── "
        tree.append(f"{prefix}{connector}{entry}")
        
        if os.path.isdir(path):  # Recursively add subdirectories
            extension = "    " if is_last else "│   "
            tree.extend(generate_tree(path, prefix + extension))
    
    return tree

repo_root = "/home/sonja/Desktop/views-platform/views-models/models/bittersweet_symphony"
repo_structure = "\n".join(generate_tree(repo_root))
repo_structure
formatted_structure = f"```\n{repo_structure}\n```"
formatted_structure


updated_readme = content.replace("## Repository Structure",
        f"## Repository Structure\n\n{formatted_structure}",
    )
updated_readme



# Write the updated content to README.md
with open(readme_path, "w") as file:
    file.write(updated_readme)