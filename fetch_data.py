import json, os
import requests
import typing
from bs4 import BeautifulSoup
import re

def load_json_file(filename: str) -> list[dict[str, typing.Any]]:
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

class Error:
    def __init__(self, message: str):
        self.message = message


def download_html(url) -> typing.Union[str, Error]:
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Save the content to a file
            return response.text
        else:
            return Error(f"Failed to retrieve the page. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        return Error(f"An error occurred: {e}")

def request_completion_from_local_llm(prompt: str) -> dict[str, str]:
    pass

def filter_html(html: str) -> typing.Union[str, Error]:
    # Parse the HTML document
    soup = BeautifulSoup(html, 'html.parser')

    # Find the <body> tag
    body = soup.find('body')
    if body != None:
        # Get the text content from the body, stripping all HTML tags
        body_text = body.get_text()
        # Remove all newline characters and leading/trailing whitespace

        # Find the keyword and get the index
        keyword = "item"
        index = body_text.lower().find(keyword)
        # Check if the keyword exists
        if index != -1:
            # Cut off all text before the keyword
            filtered_text = body_text[index:]
        else:
            filtered_text = body_text  # if keyword not found, return the whole text
        
        return filtered_text
    else:
        print(f"Body was not found for 8-k #")
        return Error(f"Body was not found for 8-k #")

# Example usage
def example(): 
    filename = 'test.eightks.json'
    text_list = []
    data = load_json_file(filename)
    for i, listing in enumerate(data):
        url = listing['files'][0]['url']
        html = download_html(url)
        if type(html) == Error:
            print(f"Error downloading html for {url} with message\n: {html.message}")
        else:
            assert type(html) == str
            filtered_text = filter_html(html)
            if type(filtered_text) != Error:
                text_list.append(filtered_text)

    with open("output.json", 'w', encoding='utf-8') as file:
        json.dump(text_list, file, indent=4)

