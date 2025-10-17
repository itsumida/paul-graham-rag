#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


INDEX_URL = "https://www.paulgraham.com/articles.html"
BASE_URL = "https://www.paulgraham.com/"


@dataclass
class EssayLink:
    title: str
    url: str
    slug: str


def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {"User-Agent": "pg-essays-scraper/1.0 (+github:local)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_index(html: str) -> List[EssayLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[EssayLink] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        # Only consider relative .html links pointing to essays
        if not href.lower().endswith(".html"):
            continue
        if urlparse(href).netloc:
            continue
        url = urljoin(BASE_URL, href)
        title = a.get_text(strip=True) or href
        links.append(EssayLink(title=title, url=url, slug=href))
    # Deduplicate by slug while preserving order
    seen = set()
    unique: List[EssayLink] = []
    for e in links:
        if e.slug not in seen:
            seen.add(e.slug)
            unique.append(e)
    return unique


def extract_title_and_text(html: str) -> tuple[str, str]:
    """Extract a reasonable title and main text from a PG essay page.

    The site is old-school HTML; essays are generally in <font> or <p> tags.
    We'll heuristically:
      - Title: use first <title> if present; else first bold <b> or first h1/h2
      - Text: concatenate visible <p> and <br>-separated text within main content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title extraction
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        b = soup.find("b")
        if b and b.get_text(strip=True):
            title = b.get_text(strip=True)
    if not title:
        for h in soup.find_all(["h1", "h2", "h3"]):
            t = h.get_text(strip=True)
            if t:
                title = t
                break
    title = title or "Untitled"

    # Remove scripts/styles/nav junk
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Heuristic: main body often inside <table> -> <td> or top-level body
    body = soup.body or soup

    # Collect text from paragraphs and line breaks
    parts: List[str] = []
    for p in body.find_all(["p", "blockquote", "pre"]):
        text = p.get_text(" ", strip=True)
        if text:
            parts.append(text)
    # Fallback: if no <p>, get text from body with minimal collapsing
    if not parts:
        text = body.get_text(" ", strip=True)
        parts = [text] if text else []

    text = "\n\n".join(parts)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\s*\n\s*)+", "\n\n", text).strip()

    return title, text


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "essay"


def write_json(out_dir: str, url: str, title: str, text: str, slug_hint: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = slug_hint or sanitize_filename(title) or "essay"
    if not slug.endswith(".json"):
        slug = f"{slug}.json"
    out_path = os.path.join(out_dir, slug)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "text": text}, f, ensure_ascii=False, indent=2)
    return out_path


def scrape_all(out_dir: str, delay_sec: float = 0.5, limit: Optional[int] = None) -> None:
    index_html = fetch_html(INDEX_URL)
    links = parse_index(index_html)
    if limit is not None:
        links = links[:limit]

    for i, link in enumerate(links, 1):
        html = fetch_html(link.url)
        title, text = extract_title_and_text(html)
        # Prefer using slug from link for stable filenames
        slug = sanitize_filename(os.path.splitext(link.slug)[0]) + ".json"
        out_path = write_json(out_dir, link.url, title, text, slug_hint=slug)
        print(f"[{i}/{len(links)}] Saved {link.title} -> {out_path}")
        time.sleep(delay_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Paul Graham essays into JSON files")
    parser.add_argument("--out", default="pg_essays_json", help="Output directory for JSON files")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of essays (for testing)")
    args = parser.parse_args()

    scrape_all(args.out, delay_sec=args.delay, limit=args.limit)


if __name__ == "__main__":
    main()


