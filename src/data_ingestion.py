# This script executes a HTTP request to the Amazon API (e.g. US products) and saves it

from api_client import *
from typing import Union
import json

QUERIES_PATH = "../api_queries.json"

def run_http_request(query: dict) -> None:   
    """
        Pull Data From API.
    """

    method = "GET"
    #url = "/search?query=Tenis&country=US" # TODO: Update the query, add other countries
    
    api_connection = ApiClient()
    for url in query["endpoints"]:
        api_connection.http_request(
                method=method, 
                request_url=url, 
                country_name=query["country_name"]
            )
    
    api_connection.save_response(filename=query["filename"]) 
    return None

def main() -> None:
    
    # Get the HTTP reponses to indicate what the API to bring us
    with open(QUERIES_PATH, "r") as f:
        queries = json.load(f)
        
    # Run # HTTP responses
    for query in queries:
        run_http_request(query)
        
    return None

if __name__=="__main__":
    main()