import unittest
import urllib.error
from unittest.mock import patch

from resource_discovery import build_direct_topic_url, discover_best_resource, find_curated_exact_resource, infer_topic_family


class ResourceDiscoveryTests(unittest.TestCase):
    @patch("resource_discovery._search_html")
    def test_direct_fallback_uses_topic_targeted_link_not_search_engine(self, mock_search_html) -> None:
        mock_search_html.side_effect = urllib.error.URLError("network unavailable")

        result = discover_best_resource("Python official documentation tutorial")

        self.assertFalse(result["live"])
        self.assertEqual(result["url"], "https://docs.python.org/3/tutorial/")
        self.assertNotIn("duckduckgo.com", result["url"])
        self.assertTrue(result.get("direct_fallback"))

    @patch("resource_discovery._search_html")
    def test_generic_fallback_does_not_emit_search_engine_url(self, mock_search_html) -> None:
        mock_search_html.side_effect = urllib.error.URLError("network unavailable")

        result = discover_best_resource("some unknown niche topic query")

        self.assertFalse(result["live"])
        self.assertTrue(result["url"].startswith("https://"))
        self.assertNotIn("duckduckgo.com", result.get("url", ""))

    @patch("resource_discovery._parse_results")
    @patch("resource_discovery._search_html")
    def test_low_quality_paid_result_is_rejected(self, mock_search_html, mock_parse_results) -> None:
        mock_search_html.return_value = "<html></html>"
        mock_parse_results.return_value = [
            {"title": "Backend bootcamp pricing plans", "url": "https://example.com/pricing"}
        ]

        result = discover_best_resource("backend bootcamp free tutorial")

        self.assertFalse(result["live"])
        self.assertNotIn("/pricing", result.get("url", ""))
        self.assertNotIn("example.com/pricing", result.get("url", ""))

    @patch("resource_discovery._url_seems_reachable", return_value=True)
    @patch("resource_discovery._parse_results")
    @patch("resource_discovery._search_html")
    def test_prefers_specific_candidate_over_generic_landing_page(
        self,
        mock_search_html,
        mock_parse_results,
        _mock_reachable,
    ) -> None:
        mock_search_html.return_value = "<html></html>"
        mock_parse_results.return_value = [
            {"title": "React search results", "url": "https://www.youtube.com/results?search_query=react"},
            {"title": "React Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=abc123"},
        ]

        result = discover_best_resource("YouTube React tutorial free")

        self.assertTrue(result["live"])
        self.assertIn("watch?v=", result["url"])

    def test_build_direct_topic_url_generates_source_specific_search_links(self) -> None:
        github_url = build_direct_topic_url("github.com", "react auth project example")
        youtube_url = build_direct_topic_url("youtube.com", "kafka tutorial free")

        self.assertIn("github.com/search", github_url)
        self.assertIn("q=", github_url)
        self.assertIn("youtube.com/results", youtube_url)
        self.assertIn("search_query=", youtube_url)

    def test_curated_exact_resource_replaces_youtube_search_with_direct_alternative(self) -> None:
        curated = find_curated_exact_resource("YouTube React tutorial free", "YouTube")

        self.assertIsNotNone(curated)
        self.assertEqual(curated["url"], "https://react.dev/learn")
        self.assertNotIn("youtube.com/results", curated["url"])

    def test_curated_exact_resource_replaces_github_search_with_direct_alternative(self) -> None:
        curated = find_curated_exact_resource("GitHub FastAPI CRUD starter tests repository", "GitHub")

        self.assertIsNotNone(curated)
        self.assertEqual(curated["url"], "https://fastapi.tiangolo.com/tutorial/")
        self.assertNotIn("github.com/search", curated["url"])

    def test_infer_topic_family_detects_cybersecurity(self) -> None:
        family = infer_topic_family("YouTube cybersecurity fundamentals networking Linux tutorial free")

        self.assertEqual(family, "cybersecurity")

    def test_curated_exact_resource_stays_in_cybersecurity_family(self) -> None:
        curated = find_curated_exact_resource("YouTube cybersecurity fundamentals networking Linux tutorial free", "YouTube")

        self.assertIsNotNone(curated)
        self.assertIn(curated["url"], {
            "https://linuxjourney.com/",
            "https://owasp.org/www-project-top-ten/",
        })
        self.assertNotIn("developer.mozilla.org", curated["url"])


if __name__ == "__main__":
    unittest.main()
