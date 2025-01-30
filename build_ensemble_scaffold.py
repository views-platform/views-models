from build_model_scaffold import ModelScaffoldBuilder
from pathlib import Path
import logging
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.templates.ensemble import (
    template_config_deployment,
    template_config_hyperparameters,
    template_config_meta,
    template_main,
    template_run_sh,
    template_requirement_txt
)
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.ensemble import EnsemblePathManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleScaffoldBuilder(ModelScaffoldBuilder):
    """
    A builder class for creating and managing the scaffold of an ensemble model.

    Attributes:
        _model (EnsemblePathManager): Manages the paths and directories for the ensemble model.
        _subdirs (dict): Subdirectories within the model directory.
        _scripts (dict): Scripts associated with the model.

    Methods:
        __init__(ensemble_name: str):
            Initializes the EnsembleScaffoldBuilder with the given ensemble name.
        
        build_model_scripts():
            Generates the necessary configuration and main scripts for the model.
            Raises FileNotFoundError if the model directory does not exist.
    """
    def __init__(self, ensemble_name: str):
        """
        Initializes the build_ensemble_scaffold instance.

        Args:
            ensemble_name (str): The name of the ensemble to be managed.

        Attributes:
            _model (EnsemblePathManager): Manages the paths for the ensemble.
            _subdirs (dict_values): The directories associated with the ensemble.
            _scripts (dict_values): The scripts associated with the ensemble.
        """
        self._model = EnsemblePathManager(ensemble_name, validate=False)
        self._subdirs = self._model.get_directories().values()
        self._scripts = self._model.get_scripts().values()

    def build_model_scripts(self):
        """
        Generates the necessary model scripts for deployment, hyperparameters, and metadata configurations.

        This method checks if the model directory exists. If it does not, it raises a FileNotFoundError.
        It then generates the following scripts using predefined templates:
        - config_deployment.py
        - config_hyperparameters.py
        - config_meta.py
        - main.py

        Raises:
            FileNotFoundError: If the model directory does not exist.
        """
        if not self._model.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self._model.model_dir} does not exist. Please call build_model_directory() first. Aborting script generation."
            )
        template_config_deployment.generate(
            script_path=self._model.configs / "config_deployment.py"
        )
        template_config_hyperparameters.generate(
            script_path=self._model.configs / "config_hyperparameters.py",
        )
        template_config_meta.generate(
            script_path=self._model.configs / "config_meta.py",
            model_name=self._model.model_name,
        )
        template_main.generate(script_path=self._model.model_dir / "main.py")
        template_run_sh.generate(script_path=self._model.model_dir / "run.sh")
        template_requirement_txt.generate(script_path=self.requirements_path, pipeline_core_version_range=PipelineConfig().views_pipeline_core_version_range)


if __name__ == "__main__":
    model_name = str(input("Enter the name of the ensemble: "))
    while (
        not ModelPathManager.validate_model_name(model_name)
        or ModelPathManager.check_if_model_dir_exists(model_name)
        or EnsemblePathManager.check_if_model_dir_exists(model_name)
    ):
        error = "Invalid input. Please use the format 'adjective_noun' in lowercase, e.g., 'happy_kitten' that does not already exist as a model or ensemble."
        logging.error(error)
        model_name = str(input("Enter the name of the ensemble: "))
    ensemble_scaffold_builder = EnsembleScaffoldBuilder(model_name)
    ensemble_scaffold_builder.build_model_directory()
    assessment = ensemble_scaffold_builder.assess_model_directory()
    if not assessment["structure_errors"]:
        logging.info("Ensemble directory structure is complete.")
    else:
        logging.warning(f"Structure errors: {assessment['structure_errors']}")
    ensemble_scaffold_builder.build_model_scripts()
    assessment = ensemble_scaffold_builder.assess_model_scripts()
    if not assessment["missing_scripts"]:
        logging.info("All scripts have been successfully generated.")
    else:
        logging.warning(f"Missing scripts: {assessment['missing_scripts']}")
    ensemble_scaffold_builder.update_gitkeep_empty_directories()
