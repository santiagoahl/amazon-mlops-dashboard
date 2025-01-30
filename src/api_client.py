# This script saves a class built to connect to the Amazon API

import http.client
import json
import pandas as pd
import os

class ApiClient():
    def __init__(self) -> None:
        self.raw_response = None
        self.temp_response = None
        self.backup_responses = []  # If multiple queries are passed, save the responses in a single list
        self.response = None
        self.temp_request_url = None
        self.backup_request_urls = []
        self.output_dir = "../data/raw/api-calls"
        self.output_path = None

    def http_request(
            self: object, 
            method: str, 
            request_url: str,
            country_name: str
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
        
        # Save the current request
        self.temp_request_url = request_url
        self.backup_request_urls.append(request_url)
        self.temp_request_url = self.temp_request_url
        self.raw_response = data_json
        
        # Save the current response
        self.temp_response = json.loads(decoded_data)["data"]["products"]
        
            # Include country name in each product data row
        for product in self.temp_response: 
            product["country_name"] = country_name 
        
        self.backup_responses.append(
            self.temp_response
        )
        
        return None
    
    def merge_responses(self) -> dict:   
        """
        Concatenates all the HTTP responses associated with a set of HTTP requests.

        Returns:
            None:
        """
        
        # Concatenate all product responses
        query_results_merged = []
        
        for temp_response in self.backup_responses:
            query_results_merged += temp_response
        
        # Format Response
        self.response = self.raw_response
        
        # Save all queries ran
        self.response["request_url"] = self.backup_request_urls
        
        # Save the merged responses
        self.response["response"] = query_results_merged         
        return None

    def save_response(self, filename: str) -> None:   
        """
        Saves the API Response into raw folder.

        Args:
            filename (str): Name of file to save the data

        Returns:
            None
        """
        #try: 
        #    response = self.temp_response
        #except ValueError:
        #    print(
        #        f"The response with type {type(self.temp_response)} has an error." 
        #        "Try running self.http_request() first"
        #    )
                
        self.output_path = os.path.join(self.output_dir, filename + ".json")

        # Merge the results into a single response
        self.merge_responses()
        
        # Save the results
        with open(file=self.output_path, mode="w") as file:
            json.dump(obj=self.response, fp=file)
        return None