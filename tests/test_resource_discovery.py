import unittest
import urllib.error
from unittest.mock import patch

from resource_discovery import discover_best_resource


class ResourceDiscoveryTests(unittest.TestCase):
    @patch("resource_discovery._search_html")
    def test_direct_fallback_uses_domain_homepage_not_search_engine(self, mock_search_html) -> None:
        mock_search_html.side_effect = urllib.error.URLError("network unavailable")

        result = discover_best_resource("Python official documentation tutorial")

        self.assertFalse(result["live"])
        self.assertTrue(result["url"].startswith("https://docs.python.org"))
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


if __name__ == "__main__":
    unittest.main()
