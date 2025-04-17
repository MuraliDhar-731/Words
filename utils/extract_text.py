import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=' ')
    except Exception as e:
        return f"Error fetching text: {e}"

def get_text_from_upload(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        return f"Error reading file: {e}"
