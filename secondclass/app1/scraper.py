import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"}

def fetch_website_contents(url):
    # add schema if user forgets to include it
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return "Could not fetch the website contents. Please check the URL and try again."

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else "No title found"

    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'img', 'svg', 'noscript','input', 'button', 'form', 'iframe']):
        tag.decompose()  # Remove script and style tags

    text = soup.get_text(separator=' ', strip=True)
    return f"Title: {title}\n\n{text}"