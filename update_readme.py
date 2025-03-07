import importlib.util
import os
from pathlib import Path
import re

os.getcwd()
configs_dir = Path("views-models/models/blank_space/configs")
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
    clf = meta_info['model_clf']
    reg = meta_info['model_reg']

target = meta_info['depvar']
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
name = queryset_info.name


## Update old README file - For Bitter Symphony Model 
scaffold_path = Path("/home/sonja/Desktop/views-platform/views-models/README_scaffold.md")
readme_path = Path("/home/sonja/Desktop/views-platform/views-models/models/bittersweet_symphony/README.md")


# Read old README
with open(readme_path, "r") as file:
    old_readme_content = file.read()


match = re.search(r"(## Created on.*)", old_readme_content, re.DOTALL)
created_section = match.group(1).strip()
created_section

insert_position = created_section.find("##")


# Find where the '##' ends (after '##' and the next space)
end_of_heading = len("##")  # Skip the '##' part itself
new_string = created_section[:end_of_heading] + " " + 'Model' + created_section[end_of_heading:]
new_string
# Read scaffold.md content
with open(scaffold_path, "r") as file:
    content = file.read()


# Dictionary of placeholders and their replacements
replacements = {
    "{{MODEL_NAME}}": model_name,
    "{{MODEL_ALGORITHM}}": algorithm,
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

# Add created section



# Write the updated content to README.md
#with open(readme_path, "w") as file:
#    file.write(content)



print("README.md successfully generated!")