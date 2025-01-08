from pathlib import Path
import logging
from views_pipeline_core.managers.package import PackageManager
import os
from views_pipeline_core.templates.package import (
    template_example_manager,
    template_gitignore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PackageScaffoldBuilder:
    def __init__(self, package_manager: PackageManager):
        self._package_manager = package_manager

    def build_package_scaffold(self):
        try:
            self._package_manager.create_views_package()
            self._package_manager.validate_views_package()
        except Exception as e:
            logger.error(f"Error creating package scaffold: {e}")
            raise e
        

    def add_gitignore(self):
        template_gitignore.generate(self._package_manager.package_path / ".gitignore")

    def build_package_directories(self):
        if not self._package_manager.manager.exists():
            self._package_manager.manager.mkdir(parents=True, exist_ok=True)

    def build_package_scripts(self):
        template_example_manager.generate(
            self._package_manager.manager / "example_manager.py"
        )


if __name__ == "__main__":

    package_name = str(input(f"Enter the name of the package: "))
    while PackageManager.validate_package_name(package_name) == False:
        error = "Invalid input. Please use the format 'views-packagename' in lowercase, e.g., 'views-stepshifter'."
        logging.error(error)
        package_name = str(input("Enter the name of the package: "))

    package_path = str(
        input(
            f"Using the package path '{Path(os.getcwd()) / package_name}'. Press Enter to confirm or enter a new base directory: "
        )
    )
    if package_path == "":
        package_path = Path(os.getcwd()) / package_name
    else:
        while not Path(package_path).parent.exists():
            error = "Invalid input. Please enter a valid directory path."
            logging.error(error)
            package_path = str(input("Enter the base directory for the package: "))
    package_scaffold_builder = PackageScaffoldBuilder(
        PackageManager(package_path, validate=False)
    )
    package_scaffold_builder.build_package_scaffold()
    package_scaffold_builder.build_package_directories()
    package_scaffold_builder.build_package_scripts()
    package_scaffold_builder.add_gitignore()
    logging.info(f"Package scaffold created at {package_path}.")
