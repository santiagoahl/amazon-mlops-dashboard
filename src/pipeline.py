# This process execute the data ingestion, preprocessing, and inferences

from config import Location, ModelParams, ProcessConfig
from prefect import flow
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

#from src.deprecated_process import process
#from src.deprecated_run_notebook import run_notebook
#from src.train import train


@flow
def iris_flow(
    location: Location = Location(),
    process_config: ProcessConfig = ProcessConfig(),
    model_params: ModelParams = ModelParams(),
):
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
    process(location, process_config)
    train(location, model_params)
    run_notebook(location)


if __name__ == "__main__":
    pass
    #iris_flow()
