"""arXiv retrieval — the grounding stage (paperreview.ai Stage 2, for posters).

Given a poster's claims, find related prior work on arXiv so the review can speak
to novelty and contextualization with evidence, not vibes. Uses the free arXiv
API (no key, no Tavily needed).
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, max_results: int = 5, timeout: int = 30) -> list[dict]:
    """Search arXiv for a query; return [{title, authors, abstract, arxiv_id, url, year}]."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    req = urllib.request.Request(f"{ARXIV_API}?{params}",
                                 headers={"User-Agent": "posterreview-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        root = ET.fromstring(r.read())

    out = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{_ATOM}summary") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{_ATOM}published") or ""
        idurl = entry.findtext(f"{_ATOM}id") or ""
        authors = [a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")]
        out.append({
            "title": title,
            "authors": [a for a in authors if a][:6],
            "abstract": summary,
            "arxiv_id": idurl.rsplit("/", 1)[-1],
            "url": idurl,
            "year": published[:4],
        })
    return out


def retrieve_related(queries: list[str], per_query: int = 4, cap: int = 12) -> list[dict]:
    """Run several queries, dedup by arxiv_id, return up to `cap` related papers."""
    seen, results = set(), []
    for q in queries:
        try:
            papers = search_arxiv(q, max_results=per_query)
        except Exception as e:
            papers = []
            print(f"  [arxiv] query failed: {q!r} ({e})")
        for p in papers:
            key = p["arxiv_id"].split("v")[0]
            if key and key not in seen:
                seen.add(key)
                results.append(p)
        time.sleep(0.5)  # be polite to the arXiv API
        if len(results) >= cap:
            break
    return results[:cap]


if __name__ == "__main__":
    import sys
    qs = sys.argv[1:] or ["scientific poster generation multimodal", "poster readability legibility design"]
    papers = retrieve_related(qs)
    print(f"{len(papers)} related papers:\n")
    for p in papers:
        print(f"- [{p['year']}] {p['title']}")
        print(f"    {', '.join(p['authors'][:3])}{'…' if len(p['authors'])>3 else ''} | {p['url']}")
