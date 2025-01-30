# This script executes a HTTP request to the Amazon API (e.g. US products) and saves it

from api_client import *
from config import Location
from typing import Union
import json
import logging
import sys

    
# Set Logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y:%m:%d %H:%M:%S",
    filename=os.path.join(Location().root_dir, "data/logs/ingestion/output.log")
)

logger = logging.getLogger(name="Logger")
logger.setLevel(logging.INFO)

def run_http_request(query: dict) -> None:   
    """
        Pull Data From API.
    """

    method = "GET"
    #url = "/search?query=Tenis&country=US" # TODO: Update the query, add other countries
    
    logging.info("Establishing Connection to API...")
    api_connection = ApiClient()
    
    for url in query["endpoints"]:
        api_connection.http_request(
                method=method, 
                request_url=url, 
                country_name=query["country_name"]
            )
    
    api_connection.save_response(filename=query["filename"]) 
    
    return None

def data_ingestion(location: Location = Location()):
    # Get the HTTP reponses to indicate what the API to bring us
    logging.info("Loading Queries...")
    with open(location.api_queries, "r") as f:
        queries = json.load(f)
    
    logging.info("Running HTTP Requests...")
    # Run # HTTP responses
    for query in queries:
        run_http_request(query)
        
    logging.info("HTTP Requests Successfully completed. Saved in {filename}".format(query["filename"]))

if __name__=="__main__":
    data_ingestion()