import os
import re
import time

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from serpapi import GoogleSearch
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# It's good practice to download the tokenizer data once
# import nltk
# nltk.download('punkt')
cache={
    "who is create you":{
        "summary":"his name is vishal",
        "error":"nothing",
        "urls_found":"novix-chat-3.onrender.com"
    }
}
app = Flask(__name__)

# --- Helper Functions ---

def clean_text(text):
    """Cleans text by removing citations, coordinates, and extra whitespace."""
    text = re.sub(r'\[\d+\]', '', text)  # Remove [number] references
    text = re.sub(r'\d{1,3}\.\d+;\s*-?\d{1,3}\.\d+', '', text)  # Remove coordinates
    text = re.sub(r'\s+', ' ', text)  # Remove multiple newlines / whitespace

    # Remove sections that clearly belong to menus
    menu_keywords = ['Home Store Tour Dates', 'Newsletter', 'Sign up for']
    for keyword in menu_keywords:
        text = text.replace(keyword, '')
    return text.strip()


def scrape_and_clean_text(url):
    """Scrapes and cleans the visible text from a given URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, and other non-content tags
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    # Try to find Wikipedia's main content div, otherwise fallback to the body
    content_div = soup.find('div', {'id': 'bodyContent'})
    if content_div:
        text_elements = content_div.find_all(string=True)
    else:
        body = soup.find('body')
        text_elements = body.find_all(string=True) if body else []

    def is_visible(element):
        """Filters out invisible elements."""
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        return element.strip() != ""

    visible_texts = filter(is_visible, text_elements)
    full_text = "\n".join(visible_texts)
    full_text = clean_text(full_text)

    if len(full_text) < 150:
        print(f"⚠️ Not enough visible content to use from {url}.\n")
        return ""
    
    return full_text


def summarize_text(full_text):
    """Summarizes a given block of text and returns a list of sentence strings."""
    if not full_text or len(full_text) < 150:
        print("⚠️ Not enough total text to summarize.")
        return []

    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 5)  # 5 sentence summary

    # Convert the summary sentences to simple strings
    return [str(sentence) for sentence in summary]

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        topic = request.form.get('topic')
        if not topic:
            return render_template('index.html', error="Please enter a topic.")
        # cache =----

        if topic in cache:
            return render_template(
            'index.html', 
            summary=cache[topic]['summary'], 
            topic=topic, 
            error=cache[topic]['error'],
            urls_found=cache[topic]['urls_found']
        )






        # IMPORTANT: Replace with your SerpAPI key, preferably from an environment variable
        api_key = os.environ.get("SERPAPI_API_KEY", "6db77591b0687e413e8ad1a22b8b8d3257c0f8116f3f5d75cac763721087d84d")
        
        params = {
            "engine": "google",
            "q": topic,
            "api_key": api_key,
            "num": "5" # Fetch top 2 results
        }

        all_text = ""
        urls_found = []
        error_message = None

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            organic_results = results.get("organic_results", [])
            urls = [result["link"] for result in organic_results]

            if not urls:
                error_message = "No URLs found. Check your API key or try another topic."
            else:
                for url in urls:
                    page_text = scrape_and_clean_text(url)
                    if page_text:
                        all_text += page_text + "\n\n"
                        urls_found.append(url)
                    time.sleep(1) # Polite delay

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = "An API or network error occurred. Please try again later."
        
        summary_sentences = summarize_text(all_text)
        temp={}
        temp['summary']=summary_sentences
        temp['error']=error_message
        temp['urls_found']=urls_found

        cache[topic]=temp


        return render_template(
            'index.html', 
            summary=summary_sentences, 
            topic=topic, 
            error=error_message,
            urls_found=urls_found
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
