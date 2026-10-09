"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import time
import random
import re
import xml.etree.ElementTree as ET  # arXiv answers with Atom XML

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn with bounded exponential backoff for RetryableError."""
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except RetryableError as exc:
            if attempt + 1 >= attempts:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = min(cap, base * (2 ** attempt)) * (0.8 + random.random() * 0.4)
            time.sleep(min(cap, max(0.0, float(delay))))


_last_arxiv_call = 0.0

def _http_get(url, params=None, cap=30.0):
    def call():
        response = httpx.get(url, params=params, timeout=30)
        if response.status_code in (429, 500, 502, 503, 504):
            retry = response.headers.get("Retry-After")
            raise RetryableError(f"HTTP {response.status_code}", float(retry) if retry and retry.isdigit() else None)
        response.raise_for_status()
        return response
    return with_retry(call, cap=cap)

def _compact(value, limit=600):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns compact JSON records."""
    global _last_arxiv_call
    try:
        terms = re.findall(r"[A-Za-z0-9]+", query)
        if not terms:
            return "NO RESULTS"
        wait = 3.0 - (time.monotonic() - _last_arxiv_call)
        if wait > 0:
            time.sleep(wait)
        _last_arxiv_call = time.monotonic()
        response = _http_get(ARXIV_URL, {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate", "sortOrder": "descending", "max_results": max(1, min(int(max_results), 30))}, 60.0)
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = (entry.findtext("a:id", namespaces=ns) or "").rstrip("/").split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}", "published": (entry.findtext("a:published", namespaces=ns) or "")[:10], "title": _compact(entry.findtext("a:title", namespaces=ns), 300), "summary": _compact(entry.findtext("a:summary", namespaces=ns))})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
def _hf_records(items, prefer_ai=False):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = item.get("paper") or item
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}", "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10], "title": _compact(paper.get("title") or item.get("title"), 300), "summary": _compact((paper.get("ai_summary") if prefer_ai else None) or paper.get("summary") or item.get("summary")), "upvotes": paper.get("upvotes", 0) or 0, "github": paper.get("githubRepo", ""), "stars": paper.get("githubStars", 0) or 0})
    return records

def _hf_call(url, params):
    return _http_get(url, params).json()

@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Return trending Hugging Face Daily Papers as compact JSON records."""
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        records = _hf_records(_hf_call(HF_DAILY_URL, params))
        if keyword:
            records = [r for r in records if keyword.lower() in (r["title"] + " " + r["summary"]).lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic and return compact JSON records."""
    try:
        if not query.strip():
            return "NO RESULTS"
        records = _hf_records(_hf_call(HF_SEARCH_URL, {"q": query, "limit": max(1, min(int(limit), 50))}), prefer_ai=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa(tool_name: str, arguments: dict) -> str:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    url = f"{EXA_URL}?exaApiKey={api_key}" if api_key else EXA_URL

    def call():
        try:
            resp = httpx.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                timeout=30.0,
            )
        except httpx.TransportError as e:
            raise RetryableError(f"Transport error: {e}")

        if resp.status_code in (429, 500, 502, 503, 504):
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
            raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)

        resp.raise_for_status()

        result = None
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if "error" in payload:
                    raise RuntimeError(f"JSON-RPC error: {payload['error']}")
                result = payload.get("result", {})
                break

        if result is None:
            raise RuntimeError("Empty response from Exa MCP")

        meta = result.get("_meta", {})
        meta_str = json.dumps(meta).lower()
        if "rate" in meta_str or "limit" in meta_str:
            raise RetryableError("Exa rate limited", retry_after=20.0)

        if result.get("isError"):
            if "rate" in str(result).lower():
                raise RetryableError("Exa rate limited", retry_after=20.0)
            raise RuntimeError(f"Exa error: {result}")

        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        joined = "\n\n".join(texts).strip()
        lowered = joined.lower()
        if "rate limit" in lowered and ("exceeded" in lowered or "try again" in lowered or "api key" in lowered or "429" in lowered):
            raise RetryableError("Exa rate limited", retry_after=20.0)

        return joined

    def safe_error(msg: str) -> str:
        if api_key:
            msg = msg.replace(api_key, "[REDACTED]")
        return msg

    try:
        text = with_retry(call, attempts=5, cap=60.0)
        return text if text else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {safe_error(str(exc))}"


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    if not query.strip():
        return "NO RESULTS"
    obj = objective.strip() or f"Find high-quality information about {query.strip()}"
    return _call_exa("web_search_exa", {"query": query, "objective": obj, "numResults": max(1, min(int(num_results), 10))})


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    if not url.strip():
        return "NO RESULTS"
    res = _call_exa("web_fetch_exa", {"urls": [url.strip()]})
    if res.startswith("ERROR:") or res == "NO RESULTS":
        return res
    return res[:12000]


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
