import importlib.util
import os
from pathlib import Path
import re
from views_pipeline_core.managers.model import ModelManager, ModelPathManager
from views_pipeline_core.managers.ensemble import  EnsembleManager, EnsemblePathManager


## change working directory to views models - optional
base_dir = os.getcwd()
#new_dir = Path(base_dir+'/views-models')
#os.chdir(new_dir)
target_dir = Path(base_dir + "/models")
target_dir

# Update repository structure:
def generate_repo_structure(folders, scripts, model_name):
    """Generate a structured repository tree dynamically from folders and scripts."""

    root_path = Path(folders["model_dir"])  # Root directory
    tree = [model_name]  # Start with the model name

    # Sort folders to ensure correct hierarchy
    sorted_folders = sorted(folders.values(), key=lambda x: x.count("/"))  
    folder_structure = {folder: [] for folder in sorted_folders}  

    # Assign scripts to their respective folders
    for script, script_path in scripts.items():
        parent_folder = str(Path(script_path).parent)
        if parent_folder in folder_structure:
            folder_structure[parent_folder].append(script)

    # Function to recursively build the tree
    def build_tree(current_path, depth=0):
        indent = "│   " * depth
        rel_path = Path(current_path).relative_to(root_path)
        tree.append(f"{indent}├── {rel_path.name}")

        # Add scripts in the current folder
        for script in sorted(folder_structure[current_path]):
            tree.append(f"{indent}│   ├── {script}")

        # Add subfolders in order
        subfolders = [f for f in sorted_folders if Path(f).parent == Path(current_path)]
        for subfolder in subfolders:
            build_tree(subfolder, depth + 1)


    root_scripts = []
    for key, value in scripts.items():
        #print(Path(value).parent)
        if Path(value).parent==Path(root_path):
                root_scripts.append(key)

    for script in sorted(root_scripts):
        tree.append(f"├── {script}")

    # Build tree from root
    for folder in sorted_folders:
        if Path(folder).parent == root_path:  # Start from root-level folders
            build_tree(folder)

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


        #repo_root = target_dir / subfolder.name
        scripts = mpm.get_scripts()
        folders = mpm.get_directories()
        scripts["run.sh"] = folders['model_dir']+'/run.sh'
        scripts["requirements.txt"] = folders['model_dir'] +'/requirements.txt'
        repo_structure = generate_repo_structure(folders, scripts, model_name=model_name)
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

        scripts = epm.get_scripts()
        folders = epm.get_directories()
        scripts["run.sh"] = folders['model_dir']+'/run.sh'
        scripts["requirements.txt"] = folders['model_dir'] +'/requirements.txt'
        repo_structure = generate_repo_structure(folders, scripts, model_name=ens_name)
        formatted_structure = f"```\n{repo_structure}\n```"
        formatted_structure


        updated_readme = content.replace("## Repository Structure",
                f"## Repository Structure\n\n{formatted_structure}",
            )
        

        # Write the updated content to README.md
        with open(readme_path, "w") as file:
            file.write(updated_readme)

print("Readme files updated!")