import concurrent.futures
import html
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"
DEFAULT_TIMEOUT = 12
MIN_ACCEPTABLE_SCORE = 9


SOURCE_HINTS = {
    "github": {"domains": ["github.com"], "label": "GitHub"},
    "youtube": {"domains": ["youtube.com"], "label": "YouTube"},
    "freecodecamp": {"domains": ["freecodecamp.org"], "label": "freeCodeCamp"},
    "khan academy": {"domains": ["khanacademy.org"], "label": "Khan Academy"},
    "coursera": {"domains": ["coursera.org"], "label": "Coursera"},
    "mit opencourseware": {"domains": ["ocw.mit.edu"], "label": "MIT OpenCourseWare"},
    "mit ocw": {"domains": ["ocw.mit.edu"], "label": "MIT OpenCourseWare"},
    "fast.ai": {"domains": ["fast.ai"], "label": "fast.ai"},
}


OFFICIAL_DOMAIN_HINTS = {
    "postgresql": "postgresql.org",
    "python": "docs.python.org",
    "dbt": "docs.getdbt.com",
    "airflow": "airflow.apache.org",
    "spark": "spark.apache.org",
    "kafka": "kafka.apache.org",
    "bigquery": "cloud.google.com",
    "snowflake": "docs.snowflake.com",
    "scikit-learn": "scikit-learn.org",
    "sklearn": "scikit-learn.org",
    "pytorch": "pytorch.org",
    "fastapi": "fastapi.tiangolo.com",
    "docker": "docs.docker.com",
    "kubernetes": "kubernetes.io",
    "terraform": "developer.hashicorp.com",
    "prometheus": "prometheus.io",
    "grafana": "grafana.com",
    "react": "react.dev",
    "next.js": "nextjs.org",
    "typescript": "typescriptlang.org",
    "node": "nodejs.org",
    "mlflow": "mlflow.org",
}


TRUSTED_DOMAIN_WEIGHTS = {
    "developer.mozilla.org": 12,
    "docs.python.org": 12,
    "python.org": 11,
    "react.dev": 12,
    "nextjs.org": 11,
    "docs.getdbt.com": 12,
    "www.typescriptlang.org": 12,
    "typescriptlang.org": 12,
    "nodejs.org": 11,
    "fastapi.tiangolo.com": 12,
    "docs.github.com": 12,
    "github.com": 10,
    "www.youtube.com": 8,
    "youtube.com": 8,
    "freecodecamp.org": 10,
    "www.freecodecamp.org": 10,
    "ocw.mit.edu": 11,
    "coursera.org": 8,
    "www.coursera.org": 8,
    "khanacademy.org": 10,
    "www.khanacademy.org": 10,
    "fast.ai": 11,
    "airflow.apache.org": 12,
    "spark.apache.org": 12,
    "kafka.apache.org": 12,
    "dbt.getdbt.com": 12,
    "www.postgresql.org": 12,
    "postgresql.org": 12,
    "scikit-learn.org": 12,
    "pytorch.org": 12,
    "mlflow.org": 11,
    "opentelemetry.io": 11,
    "kubernetes.io": 12,
    "docs.docker.com": 12,
    "developer.hashicorp.com": 12,
    "prometheus.io": 11,
    "grafana.com": 10,
    "owasp.org": 12,
    "portswigger.net": 11,
    "www.tryhackme.com": 8,
    "tryhackme.com": 8,
    "www.hackthebox.com": 8,
    "hackthebox.com": 8,
    "learn.microsoft.com": 11,
    "aws.amazon.com": 10,
    "cloud.google.com": 10,
    "docs.snowflake.com": 11,
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "basics",
    "beginner",
    "course",
    "docs",
    "documentation",
    "example",
    "for",
    "free",
    "full",
    "getting",
    "guide",
    "latest",
    "official",
    "playlist",
    "project",
    "repository",
    "search",
    "starter",
    "the",
    "to",
    "tutorial",
    "with",
}


FREE_SIGNAL_TOKENS = {
    "free",
    "open course",
    "open-source",
    "open source",
    "audit",
    "quickstart",
    "getting started",
    "beginner",
    "hands-on",
}


PAID_SIGNAL_TOKENS = {
    "pricing",
    "plans",
    "subscription",
    "subscribe",
    "trial",
    "checkout",
    "buy",
    "purchase",
    "udemy",
    "skillshare",
    "pluralsight",
}


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        text = re.sub(r"\s+", " ", "".join(self._current_text)).strip()
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text = []


def _clean_domain(domain: str) -> str:
    return domain.lower().replace("www.", "")


def _extract_source_preferences(raw_query: str) -> Dict[str, Any]:
    lowered = raw_query.lower()
    preferred_domains: List[str] = []
    source_label = "Web"
    for token, info in SOURCE_HINTS.items():
        if token in lowered:
            preferred_domains = info["domains"]
            source_label = info["label"]
            break
    return {"domains": preferred_domains, "label": source_label}


def _prepare_query(raw_query: str) -> Tuple[str, List[str], str]:
    preferences = _extract_source_preferences(raw_query)
    preferred_domains = preferences["domains"]
    source_label = preferences["label"]
    lowered = raw_query.lower()

    if not preferred_domains and any(token in lowered for token in ["official", "documentation", "docs"]):
        for keyword, domain in OFFICIAL_DOMAIN_HINTS.items():
            if keyword in lowered:
                preferred_domains = [domain]
                source_label = "Official"
                break

    query = raw_query
    if preferred_domains and "site:" not in raw_query.lower():
        query = f"{raw_query} site:{preferred_domains[0]}"
    return query, preferred_domains, source_label


def _search_html(search_query: str, timeout: int) -> str:
    url = DUCKDUCKGO_HTML_URL.format(query=urllib.parse.quote_plus(search_query))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _unwrap_result_url(href: str) -> Optional[str]:
    if not href:
        return None
    decoded = html.unescape(href)
    parsed = urllib.parse.urlparse(decoded)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query:
        return urllib.parse.unquote(query["uddg"][0])
    if decoded.startswith("//"):
        return "https:" + decoded
    if decoded.startswith("http://") or decoded.startswith("https://"):
        return decoded
    return None


def _parse_results(page_html: str) -> List[Dict[str, str]]:
    parser = _AnchorParser()
    parser.feed(page_html)
    results: List[Dict[str, str]] = []
    seen = set()
    for href, text in parser.links:
        url = _unwrap_result_url(href)
        if not url or not text:
            continue
        lower_url = url.lower()
        if "duckduckgo.com" in lower_url:
            continue
        if lower_url in seen:
            continue
        seen.add(lower_url)
        results.append({"title": text, "url": url})
    return results


def _tokenize(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\.\+\#-]*", value.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _score_result(result: Dict[str, str], original_query: str, preferred_domains: List[str]) -> int:
    parsed = urllib.parse.urlparse(result["url"])
    domain = _clean_domain(parsed.netloc)
    title = result["title"].lower()
    path = parsed.path.lower()
    score = TRUSTED_DOMAIN_WEIGHTS.get(domain, 0)

    for preferred in preferred_domains:
        if _clean_domain(preferred) == domain:
            score += 12

    if domain.startswith("docs.") or "/docs" in path or "documentation" in title:
        score += 5
    if "tutorial" in title:
        score += 3
    if "course" in title or "playlist" in title:
        score += 2
    if "github" in domain and ("awesome" in title or "example" in title or "starter" in title):
        score += 2

    content_text = f"{title} {path} {domain}"
    if any(token in content_text for token in FREE_SIGNAL_TOKENS):
        score += 4
    if any(token in content_text for token in PAID_SIGNAL_TOKENS):
        score -= 12
    if any(token in path for token in ["/pricing", "/plans", "/subscribe", "/checkout"]):
        score -= 10

    query_tokens = _tokenize(original_query)
    haystack = content_text
    overlap = sum(1 for token in query_tokens if token in haystack)
    score += overlap * 2
    if overlap == 0:
        score -= 4
    return score


def _format_display(result: Dict[str, str], source_label: str) -> str:
    return f"{source_label}: {result['title']} | {result['url']}"


def _domain_homepage(domain: str) -> str:
    cleaned = _clean_domain(domain).strip()
    if not cleaned:
        return ""
    return f"https://{cleaned}/"


def _infer_fallback_domain(raw_query: str, source_label: str) -> str:
    lowered = raw_query.lower()
    for keyword, domain in OFFICIAL_DOMAIN_HINTS.items():
        if keyword in lowered:
            return domain
    for token, info in SOURCE_HINTS.items():
        if token in lowered and info["domains"]:
            return info["domains"][0]

    label_domain = {
        "Official": "docs.python.org",
        "GitHub": "github.com",
        "YouTube": "youtube.com",
        "freeCodeCamp": "freecodecamp.org",
        "Khan Academy": "khanacademy.org",
        "Coursera": "coursera.org",
        "MIT OpenCourseWare": "ocw.mit.edu",
        "fast.ai": "fast.ai",
    }.get(source_label, "")
    if label_domain:
        return label_domain

    # Final direct-site fallback so every resource can be launched.
    return "github.com"


def _direct_fallback_resource(
    raw_query: str,
    preferred_domains: List[str],
    source_label: str,
) -> Dict[str, Any]:
    fallback_domain = _clean_domain(preferred_domains[0]) if preferred_domains else _infer_fallback_domain(raw_query, source_label)
    fallback_url = _domain_homepage(fallback_domain) if fallback_domain else ""
    fallback_title = f"{source_label} home page" if fallback_domain else "Direct learning site"
    fallback_display = (
        f"Direct: {fallback_title} | {fallback_url}"
        if fallback_url
        else "Direct: learning resource home page"
    )
    return {
        "query": raw_query,
        "title": fallback_title,
        "url": fallback_url,
        "display": fallback_display,
        "source_label": source_label if fallback_domain else "Direct",
        "live": False,
        "direct_fallback": bool(fallback_url),
        "score": 0,
    }


def discover_best_resource(raw_query: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    search_query, preferred_domains, source_label = _prepare_query(raw_query)
    try:
        page_html = _search_html(search_query, timeout)
        candidates = _parse_results(page_html)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        candidates = []

    ranked: List[Tuple[int, Dict[str, str]]] = []
    for candidate in candidates:
        ranked.append((_score_result(candidate, raw_query, preferred_domains), candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if ranked:
        best_score, best = ranked[0]
        if best_score >= MIN_ACCEPTABLE_SCORE:
            return {
                "query": raw_query,
                "title": best["title"],
                "url": best["url"],
                "display": _format_display(best, source_label),
                "source_label": source_label,
                "live": True,
                "direct_fallback": False,
                "score": best_score,
            }

    return _direct_fallback_resource(raw_query, preferred_domains, source_label)


def discover_best_resources(
    raw_queries: List[str],
    timeout: int = DEFAULT_TIMEOUT,
    max_workers: int = 6,
) -> Dict[str, Dict[str, Any]]:
    unique_queries = list(dict.fromkeys(raw_queries))
    results: Dict[str, Dict[str, Any]] = {}
    if not unique_queries:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(discover_best_resource, query, timeout): query
            for query in unique_queries
        }
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            try:
                results[query] = future.result()
            except Exception:
                _, preferred_domains, source_label = _prepare_query(query)
                results[query] = _direct_fallback_resource(query, preferred_domains, source_label)
    return results
