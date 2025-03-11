import importlib.util
import os
from pathlib import Path
import re
from views_pipeline_core.managers.model import ModelManager, ModelPathManager
from views_pipeline_core.managers.ensemble import  EnsembleManager, EnsemblePathManager


## change working directory to views models - optional
base_dir = os.getcwd()
target_dir = Path(base_dir + "/models")
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

##############################################
####            Single Models             ####
##############################################


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

##############################################
####             Ensembles                ####
##############################################

base_dir = os.getcwd()
target_ens_dir = Path(base_dir + "/ensembles")

for subfolder in target_ens_dir.iterdir():
    if subfolder.is_dir():  # Check if it's a directory
        print(f"Model: {subfolder.name}")
        #configs_dir = Path(subfolder.name+"/configs")
        configs_dir = target_ens_dir / subfolder.name / "configs"
        ens_manager = EnsembleManager(ensemble_path=EnsemblePathManager(configs_dir))
        epm = EnsemblePathManager(configs_dir)

        ## Get Meta Info
        ens_name = ens_manager.configs['name']
        ens_name = " ".join(word.capitalize() for word in ens_name.split("_"))

        models = ens_manager.configs['models']
        models = ", ".join(models)

        target = ens_manager.configs['depvar']
        if isinstance(target, list):
            target = ", ".join(target)
        level = ens_manager.configs['level']
        try:
            metrics = ens_manager.configs['metrics']
        except KeyError:
            metrics = "No information provided"
        if isinstance(metrics, list):
            metrics = ", ".join(metrics)
        
        aggregation = ens_manager.configs['aggregation']

        ## Get deployment mode 
        deployment = ens_manager.configs['deployment_status']

        ## Update old README file - For Bitter Symphony Model 
        scaffold_path = target_ens_dir / "README_ensemble_scaffold.md"
        readme_path = target_ens_dir / subfolder.name / "README.md"

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
            "{{ENSEMBLE_NAME}}": ens_name,
            "{{MODELS}}": models,
            "{{LEVEL_OF_ANALYSIS}}": level,
            "{{TARGET}}": target,
            "{{AGGREGATION}}": aggregation,
            "{{DEPLOYMENT}}": deployment,
            "{{METRICS}}": metrics,
            "{{CREATED_SECTION}}": new_string,
        }


        # Replace placeholders in scaffold content
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)

        repo_structure = generate_repo_structure(epm.get_directories(), epm.get_scripts(), model_name=ens_name)
        formatted_structure = f"```\n{repo_structure}\n```"
        formatted_structure


        updated_readme = content.replace("## Repository Structure",
                f"## Repository Structure\n\n{formatted_structure}",
            )
        

        # Write the updated content to README.md
        with open(readme_path, "w") as file:
            file.write(updated_readme)

print("Readme files updated!")