import wandb
import warnings
from pathlib import Path
from views_activelearning.cli.utils import parse_args, validate_arguments
from views_pipeline_core.managers.log import LoggingManager
from views_activelearning.managers.model import ALModelPathManager


from views_activelearning.managers.model import ALModelManager
from views_activelearning.handlers.text import ViewsTextDataset
import pandas as pd

warnings.filterwarnings("ignore")

try:
    model_path = ALModelPathManager(Path(__file__))
    logger = LoggingManager(model_path).get_logger()
except FileNotFoundError as fnf_error:
    raise RuntimeError(
        f"File not found: {fnf_error}. Check the file path and try again."
    )
except PermissionError as perm_error:
    raise RuntimeError(
        f"Permission denied: {perm_error}. Check your permissions and try again."
    )
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    dataframe = pd.read_csv("/Users/dylanpinheiro/Desktop/views-platform/experiments/activelearning/data/pol_scored.csv").head(1000)
    dataset = ViewsTextDataset(texts=dataframe["source_article"])
    ALModelManager(model_path=model_path, dataset=dataset).execute_active_learning(args=args)

# if __name__ == "__main__":
#     print(__file__)
#     model_path = ALModelPathManager(Path(__file__))
#     model_path.view_scripts()
#     model_path.view_directories()
