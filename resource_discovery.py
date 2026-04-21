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
MIN_ACCEPTABLE_SCORE = 7
MAX_VALIDATION_CANDIDATES = 6


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


CURATED_EXACT_RESOURCES = [
    {"keywords": ["freecodecamp", "responsive", "web", "design"], "title": "freeCodeCamp Responsive Web Design", "url": "https://www.freecodecamp.org/learn/2022/responsive-web-design", "source_label": "freeCodeCamp", "requires_source": ["freecodecamp"], "topic_family": "frontend"},
    {"keywords": ["freecodecamp", "javascript", "algorithms"], "title": "freeCodeCamp JavaScript Algorithms and Data Structures", "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures-v8", "source_label": "freeCodeCamp", "requires_source": ["freecodecamp"], "topic_family": "frontend"},
    {"keywords": ["freecodecamp", "sql"], "title": "freeCodeCamp Relational Database", "url": "https://www.freecodecamp.org/learn/relational-database", "source_label": "freeCodeCamp", "requires_source": ["freecodecamp"], "topic_family": "data"},
    {"keywords": ["freecodecamp", "scientific", "computing", "python"], "title": "freeCodeCamp Scientific Computing with Python", "url": "https://www.freecodecamp.org/learn/scientific-computing-with-python", "source_label": "freeCodeCamp", "requires_source": ["freecodecamp"], "topic_family": "ml"},
    {"keywords": ["freecodecamp", "data", "analysis", "python"], "title": "freeCodeCamp Data Analysis with Python", "url": "https://www.freecodecamp.org/learn/data-analysis-with-python", "source_label": "freeCodeCamp", "requires_source": ["freecodecamp"], "topic_family": "data_analysis"},
    {"keywords": ["postgresql", "sql"], "title": "PostgreSQL Tutorial", "url": "https://www.postgresql.org/docs/current/tutorial.html", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["python"], "title": "Python Tutorial", "url": "https://docs.python.org/3/tutorial/", "source_label": "Official documentation", "topic_family": "generic"},
    {"keywords": ["dbt"], "title": "dbt Docs Introduction", "url": "https://docs.getdbt.com/docs/introduction", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["airflow"], "title": "Apache Airflow Fundamentals Tutorial", "url": "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/fundamentals.html", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["spark", "pyspark"], "title": "Apache Spark Quick Start", "url": "https://spark.apache.org/docs/latest/quick-start.html", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["kafka"], "title": "Apache Kafka Quickstart", "url": "https://kafka.apache.org/quickstart", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["bigquery"], "title": "BigQuery Introduction", "url": "https://cloud.google.com/bigquery/docs/introduction", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["snowflake"], "title": "Snowflake Getting Started", "url": "https://docs.snowflake.com/en/user-guide-getting-started", "source_label": "Official documentation", "topic_family": "data"},
    {"keywords": ["html", "css", "responsive", "semantic"], "title": "MDN Learn Web Development", "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["javascript"], "title": "MDN JavaScript Guide", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["typescript"], "title": "TypeScript Handbook", "url": "https://www.typescriptlang.org/docs/handbook/intro.html", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["react"], "title": "React Learn", "url": "https://react.dev/learn", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["next.js", "nextjs"], "title": "Next.js Learn", "url": "https://nextjs.org/learn", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["fetch api", "fetch"], "title": "MDN Using Fetch", "url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["accessibility", "wcag", "wai"], "title": "WAI Tutorials", "url": "https://www.w3.org/WAI/tutorials/", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["playwright"], "title": "Playwright Introduction", "url": "https://playwright.dev/docs/intro", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["performance", "web vitals"], "title": "web.dev Learn Performance", "url": "https://web.dev/learn/performance/", "source_label": "Official documentation", "topic_family": "frontend"},
    {"keywords": ["http", "rest", "status codes"], "title": "MDN HTTP Overview", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["fastapi"], "title": "FastAPI Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["node", "express"], "title": "Introduction to Node.js", "url": "https://nodejs.org/en/learn/getting-started/introduction-to-nodejs", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["pytest"], "title": "pytest Fixtures", "url": "https://docs.pytest.org/en/stable/how-to/fixtures.html", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["opentelemetry"], "title": "What is OpenTelemetry", "url": "https://opentelemetry.io/docs/what-is-opentelemetry/", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["redis"], "title": "Redis Get Started", "url": "https://redis.io/docs/latest/develop/get-started/", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["rabbitmq"], "title": "RabbitMQ Tutorials", "url": "https://www.rabbitmq.com/tutorials", "source_label": "Official documentation", "topic_family": "backend"},
    {"keywords": ["pandas"], "title": "pandas Getting Started Tutorials", "url": "https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["numpy"], "title": "NumPy Quickstart", "url": "https://numpy.org/doc/stable/user/quickstart.html", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["scikit-learn", "sklearn"], "title": "scikit-learn Getting Started", "url": "https://scikit-learn.org/stable/getting_started.html", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["khan academy", "statistics", "probability", "hypothesis"], "title": "Khan Academy Statistics and Probability", "url": "https://www.khanacademy.org/math/statistics-probability", "source_label": "Khan Academy", "topic_family": "ml"},
    {"keywords": ["google machine learning crash course"], "title": "Google Machine Learning Crash Course", "url": "https://developers.google.com/machine-learning/crash-course", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["pytorch"], "title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["fast.ai"], "title": "Practical Deep Learning for Coders", "url": "https://course.fast.ai/", "source_label": "fast.ai", "topic_family": "ml"},
    {"keywords": ["mlflow"], "title": "MLflow Documentation", "url": "https://mlflow.org/docs/latest/index.html", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["evidently"], "title": "Evidently Documentation", "url": "https://docs.evidentlyai.com/", "source_label": "Official documentation", "topic_family": "ml"},
    {"keywords": ["owasp"], "title": "OWASP Top 10", "url": "https://owasp.org/www-project-top-ten/", "source_label": "Official documentation", "topic_family": "cybersecurity"},
    {"keywords": ["portswigger"], "title": "PortSwigger Web Security Academy", "url": "https://portswigger.net/web-security", "source_label": "Official documentation", "topic_family": "cybersecurity"},
    {"keywords": ["linux journey", "linux", "networking"], "title": "Linux Journey", "url": "https://linuxjourney.com/", "source_label": "Free web resource", "topic_family": "cybersecurity"},
    {"keywords": ["elastic", "siem"], "title": "Elastic Security Getting Started", "url": "https://www.elastic.co/guide/en/security/current/getting-started.html", "source_label": "Official documentation", "topic_family": "cybersecurity"},
    {"keywords": ["aws", "iam", "identity", "least privilege", "cloud security"], "title": "AWS IAM Getting Started", "url": "https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started.html", "source_label": "Official documentation", "topic_family": "cybersecurity"},
    {"keywords": ["incident response", "nist", "vulnerability"], "title": "NIST Incident Response Guide", "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf", "source_label": "Official documentation", "topic_family": "cybersecurity"},
    {"keywords": ["power bi"], "title": "Microsoft Learn Power BI", "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi", "source_label": "Official documentation", "topic_family": "data_analysis"},
    {"keywords": ["tableau"], "title": "Tableau Public Discover", "url": "https://public.tableau.com/app/discover", "source_label": "Free web resource", "topic_family": "data_analysis"},
    {"keywords": ["github actions"], "title": "GitHub Actions Quickstart", "url": "https://docs.github.com/en/actions/writing-workflows/quickstart", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["docker"], "title": "Docker Get Started", "url": "https://docs.docker.com/get-started/", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["terraform"], "title": "Terraform Tutorials", "url": "https://developer.hashicorp.com/terraform/tutorials", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["kubernetes"], "title": "Kubernetes Basics", "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["prometheus"], "title": "Prometheus Getting Started", "url": "https://prometheus.io/docs/prometheus/latest/getting_started/", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["grafana"], "title": "Grafana Tutorials", "url": "https://grafana.com/tutorials/", "source_label": "Official documentation", "topic_family": "devops"},
    {"keywords": ["sre"], "title": "Google SRE Workbook", "url": "https://sre.google/workbook/table-of-contents/", "source_label": "Free web resource", "topic_family": "devops"},
]


SOURCE_LABEL_FALLBACKS = {
    "Official": {"title": "Python Tutorial", "url": "https://docs.python.org/3/tutorial/", "source_label": "Official documentation"},
    "GitHub": {"title": "GitHub Actions Quickstart", "url": "https://docs.github.com/en/actions/writing-workflows/quickstart", "source_label": "Official documentation"},
    "YouTube": {"title": "MDN Learn Web Development", "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development", "source_label": "Official documentation"},
    "freeCodeCamp": {"title": "freeCodeCamp Learn", "url": "https://www.freecodecamp.org/learn/", "source_label": "freeCodeCamp"},
    "Khan Academy": {"title": "Khan Academy Statistics and Probability", "url": "https://www.khanacademy.org/math/statistics-probability", "source_label": "Khan Academy"},
    "Coursera": {"title": "Coursera Explore", "url": "https://www.coursera.org/browse", "source_label": "Coursera"},
    "MIT OpenCourseWare": {"title": "MIT OpenCourseWare", "url": "https://ocw.mit.edu/", "source_label": "MIT OpenCourseWare"},
    "fast.ai": {"title": "Practical Deep Learning for Coders", "url": "https://course.fast.ai/", "source_label": "fast.ai"},
    "Direct": {"title": "MDN Learn Web Development", "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development", "source_label": "Official documentation"},
}


TOPIC_FAMILY_KEYWORDS = {
    "data": ["data engineering", "analytics engineer", "etl", "elt", "airflow", "dbt", "warehouse", "spark", "snowflake", "bigquery", "kafka", "sql"],
    "frontend": ["frontend", "front end", "web development", "react", "next.js", "nextjs", "javascript", "typescript", "ui engineer", "html", "css", "accessibility"],
    "backend": ["backend", "back end", "api", "server", "fastapi", "node", "express", "microservices", "rest", "http", "redis", "rabbitmq", "opentelemetry", "pytest"],
    "ml": ["machine learning", "ml", "ml engineer", "deep learning", "pytorch", "tensorflow", "mlops", "scikit-learn", "sklearn", "pandas", "numpy", "mlflow", "evidently"],
    "cybersecurity": ["cybersecurity", "security", "soc analyst", "penetration", "blue team", "red team", "siem", "incident response", "owasp", "portswigger", "linux", "iam", "least privilege", "vulnerability"],
    "data_analysis": ["data analysis", "data analyst", "business intelligence", "analytics", "tableau", "power bi", "sql analyst", "dashboard", "ab testing"],
    "devops": ["devops", "cloud", "platform engineer", "site reliability", "sre", "kubernetes", "terraform", "aws", "azure", "gcp", "docker", "prometheus", "grafana", "github actions"],
}


TOPIC_FAMILY_FALLBACKS = {
    "data": {"title": "PostgreSQL Tutorial", "url": "https://www.postgresql.org/docs/current/tutorial.html", "source_label": "Official documentation"},
    "frontend": {"title": "MDN Learn Web Development", "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development", "source_label": "Official documentation"},
    "backend": {"title": "FastAPI Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/", "source_label": "Official documentation"},
    "ml": {"title": "scikit-learn Getting Started", "url": "https://scikit-learn.org/stable/getting_started.html", "source_label": "Official documentation"},
    "cybersecurity": {"title": "OWASP Top 10", "url": "https://owasp.org/www-project-top-ten/", "source_label": "Official documentation"},
    "data_analysis": {"title": "Khan Academy Statistics and Probability", "url": "https://www.khanacademy.org/math/statistics-probability", "source_label": "Khan Academy"},
    "devops": {"title": "Docker Get Started", "url": "https://docs.docker.com/get-started/", "source_label": "Official documentation"},
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


def _compact_query_for_url(raw_query: str, max_terms: int = 10) -> str:
    tokens = _tokenize(raw_query)
    if tokens:
        return " ".join(tokens[:max_terms])
    words = re.findall(r"[a-zA-Z0-9\+\#\.-]+", str(raw_query or ""))
    return " ".join(words[:max_terms]).strip()


def infer_topic_family(raw_query: str) -> str:
    lowered = str(raw_query or "").lower()
    best_family = "generic"
    best_score = 0
    for family, keywords in TOPIC_FAMILY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in lowered:
                score += max(2, len(keyword.split()))
        if score > best_score:
            best_score = score
            best_family = family
    return best_family


def find_curated_exact_resource(raw_query: str, source_label: str = "") -> Optional[Dict[str, Any]]:
    lowered = str(raw_query or "").lower()
    source_lower = str(source_label or "").lower()
    topic_family = infer_topic_family(raw_query)
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0

    for entry in CURATED_EXACT_RESOURCES:
        required_sources = [item.lower() for item in entry.get("requires_source", [])]
        if required_sources and not any(token in lowered or token in source_lower for token in required_sources):
            continue
        entry_family = str(entry.get("topic_family") or "generic")
        if topic_family != "generic" and entry_family not in {topic_family, "generic"}:
            continue
        score = 0
        for keyword in entry["keywords"]:
            if keyword in lowered:
                score += max(2, len(keyword.split()))
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match:
        return {
            "query": raw_query,
            "title": str(best_match["title"]),
            "url": str(best_match["url"]),
            "display": f"{best_match['source_label']}: {best_match['title']} | {best_match['url']}",
            "source_label": str(best_match["source_label"]),
            "live": False,
            "direct_fallback": True,
            "score": best_score,
            "curated_exact": True,
        }

    family_fallback = TOPIC_FAMILY_FALLBACKS.get(topic_family)
    if family_fallback:
        return {
            "query": raw_query,
            "title": str(family_fallback["title"]),
            "url": str(family_fallback["url"]),
            "display": f"{family_fallback['source_label']}: {family_fallback['title']} | {family_fallback['url']}",
            "source_label": str(family_fallback["source_label"]),
            "live": False,
            "direct_fallback": True,
            "score": 0,
            "curated_exact": True,
        }

    fallback = SOURCE_LABEL_FALLBACKS.get(source_label) or SOURCE_LABEL_FALLBACKS["Direct"]
    return {
        "query": raw_query,
        "title": str(fallback["title"]),
        "url": str(fallback["url"]),
        "display": f"{fallback['source_label']}: {fallback['title']} | {fallback['url']}",
        "source_label": str(fallback["source_label"]),
        "live": False,
        "direct_fallback": True,
        "score": 0,
        "curated_exact": True,
    }


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

    if _looks_generic_landing_page(result["url"]):
        score -= 14
    if len([part for part in path.split("/") if part]) >= 2:
        score += 2
    return score


def _format_display(result: Dict[str, str], source_label: str) -> str:
    return f"{source_label}: {result['title']} | {result['url']}"


def _domain_homepage(domain: str) -> str:
    cleaned = _clean_domain(domain).strip()
    if not cleaned:
        return ""
    return f"https://{cleaned}/"


def _domain_is(domain: str, expected: str) -> bool:
    cleaned = _clean_domain(domain)
    target = _clean_domain(expected)
    return cleaned == target or cleaned.endswith(f".{target}")


def _looks_generic_landing_page(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    domain = _clean_domain(parsed.netloc)
    path = (parsed.path or "/").lower().strip()
    query = urllib.parse.parse_qs(parsed.query)

    if path in {"", "/"}:
        return True

    generic_tokens = (
        "/search",
        "/results",
        "/topics",
        "/tag",
        "/tags",
        "/category",
        "/categories",
        "/explore",
        "/discover",
    )
    if any(token in path for token in generic_tokens):
        return True

    if set(query.keys()).issubset({"q", "query", "search_query", "search", "keyword"}):
        return True

    # Domain-specific generic pages.
    if _domain_is(domain, "github.com") and path in {"/", "/search"}:
        return True
    if _domain_is(domain, "youtube.com") and (path.startswith("/results") or path in {"/", "/feed"}):
        return True
    if _domain_is(domain, "freecodecamp.org") and path in {"/", "/news", "/news/"}:
        return True
    if _domain_is(domain, "coursera.org") and path.startswith("/search"):
        return True
    if _domain_is(domain, "khanacademy.org") and path.startswith("/search"):
        return True
    return False


def _url_seems_reachable(url: str, timeout: int) -> bool:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=max(2, min(timeout, 6))) as response:
            status = getattr(response, "status", 200)
            return 200 <= int(status) < 400
    except Exception:
        return False


def build_direct_topic_url(domain: str, query: str) -> str:
    cleaned = _clean_domain(domain).strip()
    if not cleaned:
        return ""

    trimmed_query = _compact_query_for_url(str(query or ""), max_terms=10)
    if not trimmed_query:
        return _domain_homepage(cleaned)

    encoded = urllib.parse.quote_plus(trimmed_query)

    if _domain_is(cleaned, "github.com"):
        return f"https://github.com/search?q={encoded}&type=repositories"
    if _domain_is(cleaned, "youtube.com"):
        return f"https://www.youtube.com/results?search_query={encoded}"
    if _domain_is(cleaned, "freecodecamp.org"):
        return f"https://www.freecodecamp.org/news/search/?query={encoded}"
    if _domain_is(cleaned, "khanacademy.org"):
        return f"https://www.khanacademy.org/search?page_search_query={encoded}"
    if _domain_is(cleaned, "coursera.org"):
        return f"https://www.coursera.org/search?query={encoded}"
    if _domain_is(cleaned, "ocw.mit.edu"):
        return f"https://ocw.mit.edu/search/?q={encoded}"
    if _domain_is(cleaned, "docs.python.org"):
        return f"https://docs.python.org/3/search.html?q={encoded}"
    if _domain_is(cleaned, "scikit-learn.org"):
        return f"https://scikit-learn.org/stable/search.html?q={encoded}"
    if _domain_is(cleaned, "docs.getdbt.com"):
        return f"https://docs.getdbt.com/search?q={encoded}"
    if _domain_is(cleaned, "react.dev"):
        return f"https://react.dev/search?q={encoded}"
    if _domain_is(cleaned, "nextjs.org"):
        return f"https://nextjs.org/search?q={encoded}"
    if _domain_is(cleaned, "typescriptlang.org"):
        return f"https://www.typescriptlang.org/search?q={encoded}"
    if _domain_is(cleaned, "nodejs.org"):
        return f"https://nodejs.org/en/search?q={encoded}"
    if _domain_is(cleaned, "docs.docker.com"):
        return f"https://docs.docker.com/search/?q={encoded}"
    if _domain_is(cleaned, "kubernetes.io"):
        return f"https://kubernetes.io/search/?q={encoded}"
    if _domain_is(cleaned, "developer.hashicorp.com"):
        return f"https://developer.hashicorp.com/search?q={encoded}"
    if _domain_is(cleaned, "pytorch.org"):
        return f"https://pytorch.org/search/?q={encoded}"
    if _domain_is(cleaned, "fastapi.tiangolo.com"):
        return f"https://fastapi.tiangolo.com/search/?q={encoded}"
    if _domain_is(cleaned, "postgresql.org"):
        return f"https://www.postgresql.org/search/?q={encoded}"
    if _domain_is(cleaned, "airflow.apache.org"):
        return f"https://airflow.apache.org/docs/apache-airflow/stable/search.html?q={encoded}"
    if _domain_is(cleaned, "spark.apache.org"):
        return f"https://spark.apache.org/search/?q={encoded}"
    if _domain_is(cleaned, "kafka.apache.org"):
        return f"https://kafka.apache.org/search/?q={encoded}"
    if _domain_is(cleaned, "fast.ai"):
        return f"https://www.fast.ai/search/?q={encoded}"

    return f"https://{cleaned}/search?q={encoded}"


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
    curated = find_curated_exact_resource(raw_query, source_label)
    if curated:
        return curated

    fallback_domain = _clean_domain(preferred_domains[0]) if preferred_domains else _infer_fallback_domain(raw_query, source_label)
    fallback_url = build_direct_topic_url(fallback_domain, raw_query) if fallback_domain else ""
    fallback_title = f"{source_label} topic results" if fallback_domain else "Direct learning site"
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
        validated_generic_candidate: Optional[Tuple[int, Dict[str, str]]] = None
        for score, candidate in ranked[:MAX_VALIDATION_CANDIDATES]:
            if score < MIN_ACCEPTABLE_SCORE:
                continue
            url = str(candidate.get("url") or "")
            if not url or not _url_seems_reachable(url, timeout):
                continue
            if _looks_generic_landing_page(url):
                if validated_generic_candidate is None:
                    validated_generic_candidate = (score, candidate)
                continue
            return {
                "query": raw_query,
                "title": candidate["title"],
                "url": candidate["url"],
                "display": _format_display(candidate, source_label),
                "source_label": source_label,
                "live": True,
                "direct_fallback": False,
                "score": score,
            }

        if validated_generic_candidate:
            score, candidate = validated_generic_candidate
            return {
                "query": raw_query,
                "title": candidate["title"],
                "url": candidate["url"],
                "display": _format_display(candidate, source_label),
                "source_label": source_label,
                "live": True,
                "direct_fallback": False,
                "score": score,
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
