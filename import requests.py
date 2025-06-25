import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import textwrap

def load_summarizer():
    """Load the BART summarization pipeline."""
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"Error loading summarizer: {e}")
        return None

def fetch_article_text(url):
    """Fetch article content from a URL using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(para.get_text() for para in paragraphs)
        return text.strip()
    except Exception as e:
        return f"Error fetching article: {e}"

def split_text(text, max_chunk_length=1000):
    """Split long text into smaller chunks for summarization."""
    return [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

def summarize_text(text, max_length=200, min_length=50):
    """Summarize the input text using a transformer model."""
    summarizer = load_summarizer()
    if summarizer is None:
        return "Error: Summarizer not available."

    try:
        chunks = split_text(text)
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) < min_length:
                continue
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return ' '.join(summaries)
    except Exception as e:
        return f"Error summarizing text: {e}"

def main():
    """Main function to run the summarizer."""
    try:
        url = input("Enter the URL of the article: ").strip()
        print("\nFetching article...\n")
        article = fetch_article_text(url)
        if not article or article.startswith("Error"):
            print(article)
            return

        print("Original Article (truncated):")
        print(textwrap.fill(article[:1000], width=100))

        print("\nGenerating Summary...\n")
        summary = summarize_text(article)
        print("Summary:")
        print(textwrap.fill(summary, width=100))
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
