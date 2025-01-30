# This process execute the data ingestion, preprocessing, and inferences

from config import Location, ModelParams, ProcessConfig
from prefect import flow
import warnings
import logging
import os
import sys
from data_ingestion import *
from data_preprocessing import *
from train import *
from predict import *

#warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

#from src.deprecated_process import process
#from src.deprecated_run_notebook import run_notebook
#from src.train import train

# Set Logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = "%(asctime)s %(levelname)s:%(name)s: %(message)s"
log_datefmt = "%Y:%m:%d %H:%M:%S"

log_file = os.path.join(Location().root_dir, "data/logs/train/output.log")
logging.basicConfig(
    format=log_format,
    level=logging.INFO,
    datefmt=log_datefmt,
    filename=log_file
)

logger = logging.getLogger(name="Logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Puedes ajustarlo a DEBUG si quieres más detalles
console_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))

logger.addHandler(console_handler)

#flow
def tennis_demmand_flow(
        location: Location = Location(),
        process_config: ProcessConfig = ProcessConfig(),
        model_params: ModelParams = ModelParams(),
    ) -> None:
    """Flow to run the process, train, and run_notebook flows

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    model_params : ModelParams, optional
        Configurations for training models, by default ModelParams()
    """
    
    #data_ingestion()
    logger.info("Running Script: data_preprocessing.py...")
    data_preprocessing()
    logger.info("Running Script: train.py...")
    train()
    logger.info("Running Script: predict.py...")
    predict()


if __name__ == "__main__":
    tennis_demmand_flow()
