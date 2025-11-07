import os
import re
import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
# FIX: Use the top-level 'search' function from serpapi for robustness
from serpapi import search 
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# --- NLTK Data Download (CRITICAL for Sumy) ---
import nltk
try:
    # Check if 'punkt' tokenizer data is available
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    # If not found, download it (this fixes a common Sumy/NLTK issue on deployment)
    nltk.download('punkt')

# --- Simple cache to reduce repeated API calls ---
cache = {
    "who is create you": {
        "summary": ["His name is Vishal"],
        "error": None,
        "urls_found": ["novix-chat-3.onrender.com"]
    }
}

app = Flask(__name__)

# --- Helper Functions ---
def clean_text(text):
    """Removes common unwanted elements and cleans whitespace."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\d{1,3}\.\d+;\s*-?\d{1,3}\.\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    for keyword in ['Home Store Tour Dates', 'Newsletter', 'Sign up for']:
        text = text.replace(keyword, '')
    return text.strip()

def scrape_and_clean_text(url):
    """Fetches, cleans, and filters text content from a given URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}, timeout=8)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header", ".ad", ".sidebar", "aside"]):
        tag.decompose()

    main_content = soup.find('main') or soup.body
    if not main_content:
        # Fallback to entire content if main tag is missing
        text_elements = soup.find_all(string=True)
    else:
        text_elements = main_content.find_all(string=True)

    def is_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        return element.strip() != ""

    visible_texts = filter(is_visible, text_elements)
    full_text = "\n".join(visible_texts)
    full_text = clean_text(full_text)

    if len(full_text) < 150:
        print(f"⚠️ Not enough relevant content from {url}. Length: {len(full_text)}")
        return ""
    return full_text

def summarize_text(full_text):
    """Generates an LSA summary of the provided text."""
    if not full_text or len(full_text) < 150:
        return []
    
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    
    try:
        summary = summarizer(parser.document, 5) # Generate 5 sentences
    except LookupError:
        # This should ideally be caught by the initial NLTK download, but included as a safeguard.
        print("⚠️ NLTK data (punkt) not found.")
        return ["Error: NLTK data missing. Cannot summarize."]
        
    return [str(sentence) for sentence in summary]

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        topic = request.form.get('topic')
        if not topic:
            return render_template('index.html', error="Please enter a topic.")

        if topic in cache:
            cached_data = cache[topic]
            return render_template(
                'index.html',
                summary=cached_data.get('summary'),
                topic=topic,
                error=cached_data.get('error'),
                urls_found=cached_data.get('urls_found')
            )

        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return render_template('index.html', error="SERPAPI_API_KEY environment variable is not set. Please set it in Render's environment settings.")

        # Updated parameters for the robust serpapi.search function
        params = {
            "engine": "google",
            "q": topic,
            "api_key": api_key,
            "num": "5" 
        }

        all_text = ""
        urls_found = []
        error_message = None

        try:
            # FIX: Use the 'search' function directly from the serpapi library
            results = search(params)
            
            organic_results = results.get("organic_results", [])
            urls = [result["link"] for result in organic_results if "link" in result]

            if not urls:
                error_message = "No relevant search results found. Try another topic."
            else:
                for url in urls:
                    page_text = scrape_and_clean_text(url)
                    if page_text:
                        all_text += page_text + "\n\n"
                        urls_found.append(url)
                    time.sleep(1)

        except Exception as e:
            print(f"⚠️ API or network error: {e}")
            error_message = f"An API or network error occurred: {e}"

        summary_sentences = summarize_text(all_text)
        
        if not summary_sentences and not error_message:
             error_message = "Could not extract enough content to generate a summary from any of the search results."
        
        cache[topic] = {
            "summary": summary_sentences,
            "error": error_message,
            "urls_found": urls_found
        }

        return render_template(
            'index.html',
            summary=summary_sentences,
            topic=topic,
            error=error_message,
            urls_found=urls_found
        )

    return render_template('index.html')

# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)