# This script executes a HTTP request to the Amazon API (e.g. US products) and saves it

import http.client
import json
import os

class ApiClient():
    def __init__(self) -> None:
        self.response = None
        self.output_dir = "../data/api-calls"
        self.output_path = None

    def http_request(
            self, 
            method, 
            request_url
        ) -> dict:
        """
        Executes a HTTP Request to server through the Amazon API

        Args:
            request_header (str): HTTP Request Header
            method (str): HTTP Method
            request_url (str): HTTP URL
            output_dir (str): Indicates where to save the results

        Returns:
            None

        """
        conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

        headers = {
            'x-rapidapi-key': "dfa8842b83msha2bc48dbc5792bdp1cbbd0jsn0bde68569041",
            'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"            
        }

        conn.request(method=method, url=request_url, headers=headers)

        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")

        data_json = {
            "method": method,
            "request_url": request_url,
            "response": decoded_data
        }
        
        self.response = data_json
        
        return data_json

    def save_response(self, filename: str) -> None:   
        """
        Saves the API Response into raw folder.

        Args:
            response (str)

        Returns:
            None
        """
        try: 
            response = self.response
        except ValueError:
            print(
                f"The response with type {type(self.response)} has an error." 
                "Try running self.http_request() first"
            )
                
        output_path = os.path.join(self.output_dir, filename + ".json")

        # Save the results
        with open(file=output_path, mode="w") as file:
            json.dump(obj=response, fp=file)
        return None
        
def main() -> None:   
    """
        Pull Data From API.
    """
    method="GET"
    url="/product-category-list?country=US"
    
    api_connection = ApiClient()
    api_connection.http_request(method=method, request_url=url)
    api_connection.save_response(filename="us_product_list")
    return None

if __name__=="__main__":
    main()
