import wandb
import warnings
from pathlib import Path
from views_activelearning.cli.utils import parse_args, validate_arguments
from views_pipeline_core.managers.log import LoggingManager
from views_activelearning.managers.model import ALModelPathManager
from views_pipeline_core.files.utils import read_dataframe
import subprocess
from views_activelearning.managers.model import ALModelManager
from views_activelearning.handlers.text import ViewsTextDataset
import os
from functools import partial
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

def start_doccano_server():
    """Start local Doccano instance using docker-compose"""
    try:
        subprocess.run([
            "docker-compose", "-f", "docker-compose.prod.yml", 
            "up", "-d", "--build"
        ], check=True)
        print("Doccano server started successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Doccano: {e}")


if __name__ == "__main__":
    model_path = ALModelPathManager(Path(__file__))
    logger = LoggingManager(model_path).get_logger()

    wandb.login()
    args = parse_args()
    validate_arguments(args)

    if os.getenv("START_DOCCANO", "false").lower() == "true":
        start_doccano_server()
    
    # Initialize dataset with multi-label support
    PATH = ""
    dataframe = read_dataframe(PATH).head(170)

    #dataset = ViewsTextDataset(
    #    texts=dataframe["what"], 
    #    labels=None,
    #    ids=dataframe["id"]
    #)

    #dataset = ViewsTextDataset(dataframe, text_col="what", id_col="id", label_col=None, n_labels=)
    # print(dataset.ids)
    # print(dataset[50431])
    # print(dataset.other_cols.columns)

    ALModelManager(
        model_path=model_path, 
        dataset=partial(ViewsTextDataset, dataframe=dataframe, text_col="text", id_col="index", label_col=None)
    ).run(args=args)