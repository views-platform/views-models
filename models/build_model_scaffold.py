from pathlib import Path
# from utils.utils_model_naming import validate_model_name
import datetime
import logging
import sys

from views_pipeline_core.templates.model import (
    template_config_deployment,
    template_config_hyperparameters,
    template_config_queryset,
    template_config_meta,
    template_config_sweep,
    template_main,
)
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.ensemble import EnsemblePathManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelScaffoldBuilder:
    """
    A class to create and manage the directory structure and scripts for a machine learning model.

    Attributes:
        model_name (str): The name of the model for which the directory structure is to be created.
        _model (ModelPathManager): An instance of the ModelPathManager class to manage model paths.
        _subdirs (list of str): A list of subdirectories to be created within the model directory.
        _scripts (list of str): A list of script paths to be created within the model directory.
        _model_algorithm (str): The algorithm used by the model.

    Methods:
        __init__(model_name: str) -> None:
            Initializes the ModelScaffoldBuilder with the given model name and sets up paths.

        build_model_directory() -> Path:
            Creates the model directory and its subdirectories, and initializes necessary files such as README.md
            and requirements.txt.

            Returns:
                Path: The path to the created model directory.

            Raises:
                FileExistsError: If the model directory already exists.

        build_model_scripts() -> None:
            Generates the necessary configuration and main scripts for the model.

            Raises:
                FileNotFoundError: If the model directory does not exist.

        assess_model_directory() -> dict:
            Assesses the model directory by checking for the presence of expected directories.

            Returns:
                dict: A dictionary containing assessment results with two keys:
                    - 'model_dir': The path to the model directory.
                    - 'structure_errors': A list of errors related to missing directories.

        assess_model_scripts() -> dict:
            Assesses the model directory by checking for the presence of expected scripts.

            Returns:
                dict: A dictionary containing assessment results with two keys:
                    - 'model_dir': The path to the model directory.
                    - 'missing_scripts': A set of missing script paths.
    """

    def __init__(self, model_name) -> None:
        """
        Initialize a ModelDirectory object with the given model name and set up paths.

        Args:
            model_name (str): The name of the model for which directories and files are to be created.

        Returns:
            None
        """
        self._model = ModelPathManager(model_name, validate=False)
        self._subdirs = self._model.get_directories().values()
        self._scripts = self._model.get_scripts().values()
        self._model_algorithm = None

    def build_model_directory(self) -> Path:
        """
        Create the model directory and its subdirectories, and initialize necessary files such as README.md and requirements.txt.

        Returns:
            Path: The path to the created model directory.

        Raises:
            FileExistsError: If the model directory already exists.
        """
        if self._model.model_dir.exists():
            logger.info(
                f"Model directory already exists: {self._model.model_dir}. Proceeding with existing directory."
            )
        else:
            self._model.model_dir.mkdir(parents=True, exist_ok=False)
            logger.info(f"Created new model directory: {self._model.model_dir}")

        for subdir in self._subdirs:
            subdir = Path(subdir)
            if not subdir.exists():
                try:
                    subdir.mkdir(parents=True, exist_ok=True)
                    if subdir.exists():
                        logging.info(f"Created subdirectory: {subdir}")
                    else:
                        logging.error(f"Did not create subdirectory: {subdir}")
                except Exception as e:
                    logging.error(f"Error creating subdirectory: {subdir}. {e}")
            else:
                logging.info(f"Subdirectory already exists: {subdir}. Skipping.")

        # Create README.md and requirements.txt
        readme_path = self._model.model_dir / "README.md"
        with open(readme_path, "w") as readme_file:
            readme_file.write(
                f"# Model README\n## Model name: {self._model.model_name}\n## Created on: {str(datetime.datetime.now())}"
            )
        if readme_path.exists():
            logging.info(f"Created README.md: {readme_path}")
        else:
            logging.error(f"Did not create README.md: {readme_path}")

        requirements_path = self._model.model_dir / "requirements.txt"
        with open(requirements_path, "w") as requirements_file:
            requirements_file.write("# Requirements\n")
        if requirements_path.exists():
            logging.info(f"Created requirements.txt: {requirements_path}")
        else:
            logging.error(f"Did not create requirements.txt: {requirements_path}")
        return self._model.model_dir

    def build_model_scripts(self):
        if not self._model.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self._model.model_dir} does not exist. Please call build_model_directory() first. Aborting script generation."
            )
        template_config_deployment.generate(
            script_dir=self._model.configs / "config_deployment.py"
        )
        self._model_algorithm = str(
            input(
                "Enter the algorithm of the model (e.g. XGBModel, LightGBMModel, HurdleModel, HydraNet): "
            )
        )
        # self._model_manager_name = str(
        #     input(
        #         "Enter the name of your ModelManager subclass (e.g. StepshifterManager, HydranetManager. Defaults to StepshifterManager): "
        #     )
        # )
        # if not self._model_manager_name:
        #     self._model_manager_name = "StepshifterManager"
        template_config_hyperparameters.generate(
            script_dir=self._model.configs / "config_hyperparameters.py",
        )
        template_config_queryset.generate(
            script_dir=self._model.queryset_path,
            model_name=self._model.model_name,
        )
        template_config_meta.generate(
            script_dir=self._model.configs / "config_meta.py",
            model_name=self._model.model_name,
            model_algorithm=self._model_algorithm,
        )
        template_config_sweep.generate(
            script_dir=self._model.configs / "config_sweep.py",
            model_name=self._model.model_name,
            model_algorithm=self._model_algorithm,
        )
        template_main.generate(script_dir=self._model.model_dir / "main.py")
        # INFO: utils_outputs.py was not templated because it will probably be moved to common_utils in the future.
        print(f"Remember to update the queryset file at {self._model.queryset_path}!")


    def assess_model_directory(self) -> dict:
        """
        Assess the model directory by checking for the presence of expected directories.

        Returns:
            dict: A dictionary containing assessment results with two keys:
                - 'model_dir': The path to the model directory.
                - 'structure_errors': A list of errors related to missing directories or files.
        """
        assessment = {"model_dir": self._model.model_dir, "structure_errors": []}
        if not self._model.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self._model.model_dir} does not exist. Please call build_model_directory() first."
            )
        updated_model_path = ModelPathManager(self._model.model_name, validate=True)
        assessment["structure_errors"] = set(
            updated_model_path.get_directories().values()
        ) - set(self._subdirs)
        del updated_model_path
        return assessment

    def assess_model_scripts(self) -> dict:
        """
        Assess the model directory by checking for the presence of expected directories.

        Returns:
            dict: A dictionary containing assessment results with two keys:
                - 'model_dir': The path to the model directory.
                - 'structure_errors': A list of errors related to missing directories or files.
        """
        assessment = {"model_dir": self._model.model_dir, "missing_scripts": set()}
        if not self._model.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self._model.model_dir} does not exist. Please call build_model_directory() first."
            )
        for script_path in self._scripts:
            script_path = Path(script_path)
            if not script_path.exists():
                assessment["missing_scripts"].add(script_path)
        return assessment
    
    # Add a .gitkeep file to empty directories and remove it from non-empty directories
    def update_gitkeep_empty_directories(self, delete_gitkeep=True):
        for subdir in self._subdirs:
            subdir = Path(subdir)
            if not list(subdir.glob("*")):
                gitkeep_path = subdir / ".gitkeep"
                if not gitkeep_path.exists():
                    gitkeep_path.touch()
                    logging.info(f"Created .gitkeep file in empty directory: {subdir}")
            else:
                if delete_gitkeep:
                    gitkeep_path = subdir / ".gitkeep"
                    if gitkeep_path.exists():
                        gitkeep_path.unlink()
                        logging.info(f"Removed .gitkeep file from non-empty directory: {subdir}")

if __name__ == "__main__":
    model_name = str(input("Enter the name of the model: "))
    while (
        not ModelPathManager.validate_model_name(model_name)
        or ModelPathManager.check_if_model_dir_exists(model_name)
        or EnsemblePathManager.check_if_model_dir_exists(model_name)
    ):
        error = "Invalid input. Please use the format 'adjective_noun' in lowercase, e.g., 'happy_kitten' that does not already exist as a model or ensemble."
        logging.error(error)
        model_name = str(input("Enter the name of the model: "))
    model_directory_builder = ModelScaffoldBuilder(model_name)
    model_directory_builder.build_model_directory()
    assessment = model_directory_builder.assess_model_directory()
    if not assessment["structure_errors"]:
        logging.info("Model directory structure is complete.")
    else:
        logging.warning(f"Structure errors: {assessment['structure_errors']}")
    model_directory_builder.build_model_scripts()
    assessment = model_directory_builder.assess_model_scripts()
    if not assessment["missing_scripts"]:
        logging.info("All scripts have been successfully generated.")
    else:
        logging.warning(f"Missing scripts: {assessment['missing_scripts']}")
    model_directory_builder.update_gitkeep_empty_directories()
