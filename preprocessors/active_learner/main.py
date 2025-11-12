import os
import subprocess
from functools import partial
import warnings
from pathlib import Path
import wandb
from views_pipeline_core.files.utils import read_dataframe
from views_activelearning.cli.utils import parse_args, validate_arguments
from views_activelearning.managers.model import ALModelPathManager, ALModelManager
from views_activelearning.handlers.text import ViewsTextDataset


warnings.filterwarnings("ignore")

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

    wandb.login()
    args = parse_args()
    validate_arguments(args)

    if os.getenv("START_DOCCANO", "false").lower() == "true":
        start_doccano_server()
    
    # Initialize dataset with multi-label support
    # PATH = Path.home() / "edattack_synthetic_texts" / "lemonade.csv"
    # dataframe = read_dataframe('/home/sonja/Desktop/ucdp_aec_try/ucdp_aec_data.csv')
    # dataframe = read_dataframe('/home/sonja/Downloads/windows.csv')/home/sonja/Downloads/Africa_lagged_data_up_to-2024-09-12.xlsx
    # dataframe = read_dataframe('/home/sonja/Desktop/Edattack sprint/data/combined_icr_acled.csv')
    HOME = Path.home()
    file_path = HOME / 'views-platform/experiments/data/edattack_synthetic_data/combined_icr_acled.csv'
    dataframe = read_dataframe(file_path)

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
        dataset=partial(ViewsTextDataset, dataframe=dataframe, text_col="window_text", id_col="index", label_col=None) #text_col="plain_text", id_col="event_id_in_acled"
    ).run(args=args)
