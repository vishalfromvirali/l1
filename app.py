import os
import re
import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from serpapi import GoogleSearch   # ✅ Correct import for serpapi==0.1.5
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Simple in-memory cache
cache = {}

app = Flask(__name__)

# ---------------------- TEXT CLEANING ----------------------
def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------- SCRAPE WEBSITE ----------------------
def scrape_and_clean_text(url):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    text_elements = main.get_text(separator="\n")
    text = clean_text(text_elements)

    if len(text) < 150:
        return ""

    return text

# ---------------------- SUMMARIZE ----------------------
def summarize_text(text):
    if len(text) < 150:
        return ["Not enough content to summarize."]

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    try:
        sentences = summarizer(parser.document, 5)
    except Exception:
        return ["Summary error (NLTK missing)."]

    return [str(s) for s in sentences]

# ---------------------- ROUTE ----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        topic = request.form.get("topic")

        if not topic:
            return render_template("index.html", error="Please enter a topic.")

        # Cache hit
        if topic in cache:
            return render_template("index.html", **cache[topic])

        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return render_template("index.html", error="SERPAPI_API_KEY is not set.")

        params = {
            "engine": "google",
            "q": topic,
            "num": 5,
            "api_key": api_key
        }

        try:
            search = GoogleSearch(params)
            result = search.get_dict()
            organic = result.get("organic_results", [])
            urls = [i["link"] for i in organic if "link" in i]
        except Exception as e:
            return render_template("index.html", error=f"API Error: {e}")

        combined_text = ""
        urls_found = []

        for url in urls:
            txt = scrape_and_clean_text(url)
            if txt:
                combined_text += txt + "\n\n"
                urls_found.append(url)
            time.sleep(1)

        summary = summarize_text(combined_text)

        cache[topic] = {
            "summary": summary,
            "error": None,
            "urls_found": urls_found,
            "topic": topic
        }

        return render_template("index.html", **cache[topic])

    return render_template("index.html")

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
