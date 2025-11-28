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
    dataframe = read_dataframe('/home/sonja/Downloads/acled_tenthousand.csv')
    #HOME = Path.home()
    #file_path = HOME / 'views-platform/experiments/data/edattack_synthetic_data/combined_icr_acled.csv'
    #dataframe = read_dataframe(file_path)


    # --- OPTIONAL INFERENCE DATA ---
    inference_df = None
    inference_text_col = None
    inference_id_col = None

    if getattr(args, "inference", False) and getattr(args, "inference_data", None):
        inference_df = read_dataframe(args.inference_data)
        # define columns IN main.py as you wanted
        inference_text_col = args.inference_text_col or "notes"
        inference_id_col = args.inference_id_col or "event_id_cnty"

    manager = ALModelManager(
        model_path=model_path,
        dataset=partial(
            ViewsTextDataset,
            dataframe=dataframe.head(5000),
            text_col="notes",
            id_col="event_id_cnty",
            label_col=None,
        ),
    )


    manager.run(
        args=args,
        inference_df=inference_df,
        inference_text_col=inference_text_col,
        inference_id_col=inference_id_col,
    )

    

    #ALModelManager(
    #    model_path=model_path, 
    #    dataset=partial(ViewsTextDataset, dataframe=dataframe.head(5000), text_col="notes", id_col="event_id_cnty", label_col=None) #text_col="plain_text", id_col="event_id_in_acled"
    #).run(args=args)
