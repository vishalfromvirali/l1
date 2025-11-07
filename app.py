import os
import re
import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from google_search_results import GoogleSearchResults
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# --- Simple cache to reduce repeated API calls ---
cache = {
    "who is create you": {
        "summary": ["His name is Vishal"],
        "error": "nothing",
        "urls_found": ["novix-chat-3.onrender.com"]
    }
}

app = Flask(__name__)

# --- Helper Functions ---
def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\d{1,3}\.\d+;\s*-?\d{1,3}\.\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    for keyword in ['Home Store Tour Dates', 'Newsletter', 'Sign up for']:
        text = text.replace(keyword, '')
    return text.strip()

def scrape_and_clean_text(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    content_div = soup.find('div', {'id': 'bodyContent'})
    text_elements = content_div.find_all(string=True) if content_div else (soup.body.find_all(string=True) if soup.body else [])

    def is_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        return element.strip() != ""

    visible_texts = filter(is_visible, text_elements)
    full_text = "\n".join(visible_texts)
    full_text = clean_text(full_text)

    if len(full_text) < 150:
        print(f"⚠️ Not enough visible content from {url}.")
        return ""
    return full_text

def summarize_text(full_text):
    if not full_text or len(full_text) < 150:
        return []
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 5)
    return [str(sentence) for sentence in summary]

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        topic = request.form.get('topic')
        if not topic:
            return render_template('index.html', error="Please enter a topic.")

        if topic in cache:
            return render_template(
                'index.html',
                summary=cache[topic]['summary'],
                topic=topic,
                error=cache[topic]['error'],
                urls_found=cache[topic]['urls_found']
            )

        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return render_template('index.html', error="SERPAPI_API_KEY not set.")

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
            search = GoogleSearchResults(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            urls = [result["link"] for result in organic_results]

            if not urls:
                error_message = "No URLs found. Try another topic."
            else:
                for url in urls:
                    page_text = scrape_and_clean_text(url)
                    if page_text:
                        all_text += page_text + "\n\n"
                        urls_found.append(url)
                    time.sleep(1)

        except Exception as e:
            print(f"⚠️ API or network error: {e}")
            error_message = "An API or network error occurred."

        summary_sentences = summarize_text(all_text)
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
    app.run(host="0.0.0.0", port=port, debug=False)
