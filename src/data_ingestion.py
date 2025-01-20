# This script executes a HTTP request to the Amazon API (e.g. US products) and saves it

from api_client import *

def main() -> None:   
    """
        Pull Data From API.
    """
    method="GET"
    url="/product-category-list?country=US" # TODO: Update the query, add other countries
    
    api_connection = ApiClient()
    api_connection.http_request(method=method, request_url=url)
    api_connection.save_response(filename="us_product_list") 
    return None

if __name__=="__main__":
    main()