import os
import re
import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from serpapi import search     # ✅ Only working import in serpapi==0.1.5
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Cache
cache = {}

app = Flask(__name__)


def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def scrape_and_clean_text(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla"}, timeout=8)
        r.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "iframe"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    if not main:
        return ""

    texts = [t.strip() for t in main.stripped_strings]
    final = " ".join(texts)
    final = clean_text(final)

    if len(final) < 150:
        return ""

    return final


def summarize_text(full_text):
    if len(full_text) < 150:
        return []

    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 5)
    return [str(sentence) for sentence in summary]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        topic = request.form.get("topic")

        if not topic:
            return render_template("index.html", error="Please enter a topic.")

        if topic in cache:
            return render_template(
                "index.html",
                summary=cache[topic]["summary"],
                urls_found=cache[topic]["urls"],
                error=None
            )

        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return render_template("index.html", error="SERPAPI_API_KEY not set")

        # ✅ FIX: Use serpapi.search() instead of GoogleSearch
        result = search(
            q=topic,
            engine="google",
            api_key=api_key,
            num=5
        )

        urls = []
        organic = result.get("organic_results", [])

        for item in organic:
            if "link" in item:
                urls.append(item["link"])

        all_text = ""
        for u in urls:
            text = scrape_and_clean_text(u)
            if text:
                all_text += text
            time.sleep(1)

        summary = summarize_text(all_text)

        cache[topic] = {"summary": summary, "urls": urls}

        return render_template(
            "index.html",
            summary=summary,
            urls_found=urls,
            error=None
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
