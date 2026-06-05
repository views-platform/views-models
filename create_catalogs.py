import os
import importlib.util
import logging
import subprocess
import tempfile
from pathlib import Path

from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.ensemble import EnsemblePathManager

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s %(name)s - %(levelname)s - %(message)s"
)


GITHUB_URL = 'https://github.com/views-platform/views-models/blob/main/' 

# Scaffold/fixture entries that exist for testing purposes only.
_FIXTURE_ENTRIES = {"fake_model", "test_model", "test_ensemble"}


def get_implementation_date(config_meta_path, default_date="2026-01-01"):
    """Get the date when a config_meta.py was first added to git."""
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%aI", "--", str(config_meta_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            iso_date = result.stdout.strip().split('\n')[-1]
            return iso_date[:10]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return default_date



def extract_models(model_class):
    """
    It creates a dictionary containing all the necessary information about a model by merging the config_meta.py, config_deployement.py and config_hyperparameters.py dictionaries.

    Parameters:
    model_class: ModelPath class object from ModelPath.py

    Returns:
    model_dict: A dictionary containing the following relevant keys:
        -name: model name from config_meta.py
        -algorithm: algorithm from config_meta.py
        -targets: targets from config_meta.py
        -queryset: markdown link with marker 'queryset' from config_meta.py pointing to the queryset in common_querysets
        -level: 'priogrid_month' or 'country_month' from queryset
        -creator: creator from config_meta.py
        -deployment_status: deployment_status from config_deployment.py
        -hyperparameters: markdown link with marker 'hyperparameters model_name' config_meta.py pointing to the model specific config_hyperparameters.py
    """
    
    model_dict = {}
    model_dict['model_dir_path'] = Path(model_class.model_dir)
    config_meta = os.path.join(model_class.configs, 'config_meta.py')
    config_modelset = os.path.join(model_class.configs, 'config_modelset.py')
    config_deployment = os.path.join(model_class.configs, 'config_deployment.py')
    config_hyperparameters = os.path.join(model_class.configs, 'config_hyperparameters.py')

    
    if os.path.exists(config_meta):
        logging.info(f"Found meta config: {config_meta}")
        spec = importlib.util.spec_from_file_location(f"config_meta_{Path(config_meta).parent.parent.name}", config_meta)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dict.update(module.get_meta_config())
        model_dict['implementation_date'] = get_implementation_date(config_meta)
        config_queryset = os.path.join(model_class.configs, 'config_queryset.py')
        if model_class.model_name.endswith('baseline'):
            model_dict['queryset'] = 'N/A'
        elif os.path.exists(config_queryset):
            model_dict['queryset'] = create_link(f"{model_class.model_name}_features", Path(config_queryset))
        else:
            model_dict['queryset'] = 'None'


    if os.path.exists(config_modelset):
        logging.info(f"Found modelset config: {config_modelset}")
        spec = importlib.util.spec_from_file_location(f"config_modelset_{Path(config_modelset).parent.parent.name}", config_modelset)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dict.update(module.get_modelset_config())
        model_dict['modelset_link'] = create_link(
            f"{model_class.model_name}_constituent_models", Path(config_modelset)
        )

    if os.path.exists(config_deployment):
        logging.info(f"Found deployment config: {config_deployment}")
        spec = importlib.util.spec_from_file_location(f"config_deployment_{Path(config_deployment).parent.parent.name}", config_deployment)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dict.update(module.get_deployment_config())

    if os.path.exists(config_hyperparameters):
        logging.info(f"Found hyperparameters config: {config_hyperparameters}") 
        model_dict['hyperparameters'] = create_link(f"hyperparameters {model_class.model_name}", Path(model_class.get_scripts()['config_hyperparameters.py']))
    
    return model_dict



def create_link(marker, filepath: Path, prefix="- "):
    """
    Generates a markdown-formatted link to a specific file or directory in the repository's main branch.

    Parameters:
    marker: a marker that will be displayed as the clickable text in the markdown link
    filepath: absolute path of the file or directory
    prefix: string prepended to the link (default "- " for list items, use "" for table cells)

    Returns:
    str: A markdown link in the format `{prefix}[marker](GITHUB_URL/relative_filepath)`
    """
    relative_path = filepath.relative_to(ModelPathManager.get_root())
    link_template = '{prefix}[{marker}]({url}{file})'
    return link_template.format(prefix=prefix, marker=marker, url=GITHUB_URL, file=relative_path)



def _build_markdown_table(headers, rows):
    """Build a markdown table string from headers and row data."""
    markdown_table = '| ' + ' '.join([f"{header} |" for header in headers]) + '\n'
    markdown_table += '| ' + ' '.join(['-' * len(header) + ' |' for header in headers]) + '\n'
    for row in rows:
        markdown_table += '| ' + ' | '.join(row) + ' |\n'
    return markdown_table


def _format_name_cell(model):
    """Format model/ensemble name as a clickable link or plain text."""
    name = model.get('name', '')
    model_dir = model.get('model_dir_path')
    return create_link(name, model_dir, prefix="") if model_dir else name


def _format_targets(model):
    """Extract and format targets as a comma-separated string."""
    targets = model.get('targets', '') or model.get('regression_targets', '')
    if isinstance(targets, list):
        targets = ', '.join(targets)
    return targets


def generate_model_table(models_list):
    """Generate a markdown catalog table for individual models."""
    headers = ['Model Name', 'Algorithm', 'Targets', 'Input Features', 'Hyperparameters', 'Implementation Status', 'Implementation Date', 'Author']
    rows = []
    for model in models_list:
        rows.append([
            _format_name_cell(model),
            str(model.get('algorithm', '')).split('(')[0],
            _format_targets(model),
            model.get('queryset', ''),
            model.get('hyperparameters', ''),
            model.get('deployment_status', ''),
            model.get('implementation_date', ''),
            model.get('creator', ''),
        ])
    return _build_markdown_table(headers, rows)


def generate_ensemble_table(ensembles_list):
    """Generate a markdown catalog table for ensembles."""
    headers = ['Ensemble Name', 'Algorithm', 'Targets', 'Constituent Models', 'Hyperparameters', 'Implementation Status', 'Implementation Date', 'Author']
    rows = []
    for ensemble in ensembles_list:
        rows.append([
            _format_name_cell(ensemble),
            ensemble.get('aggregation', ''),
            _format_targets(ensemble),
            ensemble.get('modelset_link', ''),
            ensemble.get('hyperparameters', ''),
            ensemble.get('deployment_status', ''),
            ensemble.get('implementation_date', ''),
            ensemble.get('creator', ''),
        ])
    return _build_markdown_table(headers, rows)



def update_readme_with_tables(
    readme_path, cm_table, pgm_table, ensemble_table
):
    """
    Updates the tables in README.md between defined placeholders.

    Args:
        readme_path (str): Path to the README file.
        pgm_table (str): Markdown table for PGM models.
        cm_table (str): Markdown table for CM models.
        ensemble_table (str): Markdown table for ensembles.
    """
    with open(readme_path, "r") as file:
        content = file.read()

    content = replace_table_in_section(
        content, "PGM_TABLE", pgm_table
    )
    content = replace_table_in_section(
        content, "CM_TABLE", cm_table
    )
    content = replace_table_in_section(
        content, "ENSEMBLE_TABLE", ensemble_table
    )

    dir_name = os.path.dirname(os.path.abspath(readme_path))
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, readme_path)


def replace_table_in_section(content, section_name, new_table):
    """
    Replaces table content between placeholders in a Markdown section.

    Args:
        content (str): The original file content.
        section_name (str): Name of the placeholder section.
        new_table (str): The new table to insert.

    Returns:
        str: Updated content with the new table.
    """
    start_marker = f"<!-- {section_name}_START -->"
    end_marker = f"<!-- {section_name}_END -->"

    before, _, after = content.partition(start_marker)
    _, _, after = after.partition(end_marker)

    updated_content = (
        before + start_marker + "\n" + new_table + "\n" + end_marker + after
    )
    return updated_content















if __name__ == "__main__":
    models_list_cm = []
    models_list_pgm = []
    ensemble_list = []

    base_dirs = ["models", "ensembles"]

    for model_type in base_dirs:
        if os.path.isdir(model_type):
            for model_name in sorted(os.listdir(model_type)):
                if  ModelPathManager.validate_model_name(model_name) and model_name not in _FIXTURE_ENTRIES:
                    model_path = os.path.join(model_type, model_name)
                    if os.path.isdir(model_path):  
                        if model_type=='models':
                            model_class = ModelPathManager(model_name, validate=False)
                            model = extract_models(model_class)
                            if 'level' in model and model['level'] == 'pgm':
                                models_list_pgm.append(model)
                            elif 'level' in model and model['level'] == 'cm':
                                models_list_cm.append(model)
                        elif model_type=='ensembles':
                            ensemble_class = EnsemblePathManager(model_name, validate=False)
                            model = extract_models(ensemble_class)
                            ensemble_list.append(model)
                        


            




    markdown_table_cm = generate_model_table(models_list_cm)
    markdown_table_pgm = generate_model_table(models_list_pgm)
    markdown_table_ensembles = generate_ensemble_table(ensemble_list)

    # Update README.md file
    update_readme_with_tables(
        "README.md",
        markdown_table_cm,
        markdown_table_pgm,
        markdown_table_ensembles,
    )

