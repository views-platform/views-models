from pathlib import Path
import datetime
import logging
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.templates.model import (
    template_config_deployment,
    template_config_hyperparameters,
    template_config_queryset,
    template_config_meta,
    template_config_sweep,
    template_config_partitions,
    template_main,
    template_run_sh,
    template_requirement_txt
)
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.managers.ensemble import EnsemblePathManager
from views_pipeline_core.managers.package import PackageManager
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

    def __init__(self, model_name: str) -> None:
        """
        Initialize a ModelDirectory object with the given model name and set up paths.

        Args:
            model_name (str): The name of the model for which directories and files are to be created.

        Attributes:
            _model (ModelPathManager): An instance of ModelPathManager initialized with the given model name.
            _subdirs (dict_values): The subdirectories associated with the model.
            _scripts (dict_values): The scripts associated with the model.
            _model_algorithm (None): Placeholder for the model algorithm, initially set to None.
        """
        self._model = ModelPathManager(model_name, validate=False)
        self._subdirs = self._model.get_directories().values()
        self._scripts = self._model.get_scripts().values()
        self._model_algorithm = None
        self.package_name = None

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

        self.requirements_path = self._model.model_dir / "requirements.txt"
        # with open(requirements_path, "w") as requirements_file:
        #     requirements_file.write("# Requirements\n")
        # if requirements_path.exists():
        #     logging.info(f"Created requirements.txt: {requirements_path}")
        # else:
        #     logging.error(f"Did not create requirements.txt: {requirements_path}")
        return self._model.model_dir

    def build_model_scripts(self):
        """
        Generates various model configuration and script files required for the model.

        This method performs the following steps:
        1. Checks if the model directory exists. If not, raises a FileNotFoundError.
        2. Prompts the user to input the algorithm of the model.
        3. Generates the `config_deployment.py` script.
        4. Generates the `config_hyperparameters.py` script.
        5. Generates the queryset configuration script.
        6. Generates the `config_meta.py` script with model name and algorithm.
        7. Generates the `config_sweep.py` script with model name and algorithm.
        8. Generates the `config_partitions.py` script.
        9. Generates the main script for the model.
        10. Reminds the user to update the queryset file.

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
        self._model_algorithm = str(
            input(
                "Enter the algorithm of the model (e.g. XGBModel, LightGBMModel, HurdleModel, HydraNet): "
            )
        )
        template_config_hyperparameters.generate(
            script_path=self._model.configs / "config_hyperparameters.py",
        )
        template_config_queryset.generate(
            script_path=self._model.queryset_path,
            model_name=self._model.model_name,
        )
        template_config_meta.generate(
            script_path=self._model.configs / "config_meta.py",
            model_name=self._model.model_name,
            model_algorithm=self._model_algorithm,
        )
        template_config_sweep.generate(
            script_path=self._model.configs / "config_sweep.py",
            model_name=self._model.model_name,
            model_algorithm=self._model_algorithm,
        )
        template_config_partitions.generate(script_path=self._model.configs / "config_partitions.py")
        template_main.generate(script_path=self._model.model_dir / "main.py")

        self.package_name = str(input("Enter the name of the architecture package: "))
        while (PackageManager.validate_package_name(self.package_name) == False):
            error = "Invalid input. Please use the format 'views-packagename' in lowercase, e.g., 'views-stepshifter'."
            logging.error(error)
            self.package_name = str(input("Enter the name of the architecture package: "))
        template_run_sh.generate(script_path=self._model.model_dir / "run.sh", package_name=self.package_name)
        try:
            _latest_package_release_version = PackageManager.get_latest_release_version_from_github(self.package_name)
        except Exception as e:
            logging.error(f"Error fetching latest release version for {self.package_name}: {e}. Using default version 0.1.0.")
            _latest_package_release_version = None
        template_requirement_txt.generate(script_path=self.requirements_path, package_name=self.package_name, package_version_range=_latest_package_release_version)


        print(f"\033[91m\033[1mRemember to update the queryset file at {self._model.queryset_path}!\033[0m")

    def assess_model_directory(self) -> dict:
        """
        Assess the structure of the model directory and return any discrepancies.

        This method checks if the model directory exists and validates its structure
        based on the target type ('model' or 'ensemble'). It returns a dictionary
        containing the model directory path and any structural errors found.

        Returns:
            dict: A dictionary with the following keys:
                - "model_dir" (Path): The path to the model directory.
                - "structure_errors" (list): A list of structural errors found in the model directory.

        Raises:
            FileNotFoundError: If the model directory does not exist.
            ValueError: If the target type is invalid.
        """
        assessment = {"model_dir": self._model.model_dir, "structure_errors": []}
        if not self._model.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self._model.model_dir} does not exist. Please call build_model_directory() first."
            )
        if self._model.target == "model":
            updated_model_path = ModelPathManager(self._model.model_name, validate=True)
        elif self._model.target == "ensemble":
            updated_model_path = EnsemblePathManager(
                self._model.model_name, validate=True
            )
        else:
            raise ValueError(
                "Invalid target set in ModelPathManager: {self._model.target}."
            )
        assessment["structure_errors"] = set(
            updated_model_path.get_directories().values()
        ) - set(self._subdirs)
        del updated_model_path
        return assessment

    def assess_model_scripts(self) -> dict:
        """
        Assess the presence of model scripts in the model directory.

        This method checks if the model directory exists and verifies the presence
        of each script specified in the `_scripts` attribute. If the model directory
        does not exist, a `FileNotFoundError` is raised. The method returns a 
        dictionary containing the model directory path and a set of missing scripts.

        Returns:
            dict: A dictionary with the following keys:
                - "model_dir" (Path): The path to the model directory.
                - "missing_scripts" (set): A set of script paths that are missing.

        Raises:
            FileNotFoundError: If the model directory does not exist.
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
        """
        Updates the .gitkeep files in empty directories within the specified subdirectories.

        This method iterates over the subdirectories specified in self._subdirs. For each subdirectory:
        - If the subdirectory is empty, it creates a .gitkeep file if it does not already exist.
        - If the subdirectory is not empty and delete_gitkeep is True, it removes the .gitkeep file if it exists.

        Args:
            delete_gitkeep (bool): If True, removes .gitkeep files from non-empty directories. Default is True.

        Logs:
            - Creation of .gitkeep files in empty directories.
            - Removal of .gitkeep files from non-empty directories.
        """
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
                        logging.info(
                            f"Removed .gitkeep file from non-empty directory: {subdir}"
                        )


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
    model_scaffold_builder = ModelScaffoldBuilder(model_name)
    model_scaffold_builder.build_model_directory()
    assessment = model_scaffold_builder.assess_model_directory()
    if not assessment["structure_errors"]:
        logging.info("Model directory structure is complete.")
    else:
        logging.warning(f"Structure errors: {assessment['structure_errors']}")
    model_scaffold_builder.build_model_scripts()
    assessment = model_scaffold_builder.assess_model_scripts()
    if not assessment["missing_scripts"]:
        logging.info("All scripts have been successfully generated.")
    else:
        logging.warning(f"Missing scripts: {assessment['missing_scripts']}")
    model_scaffold_builder.update_gitkeep_empty_directories()
