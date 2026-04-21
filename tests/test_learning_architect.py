import json
import unittest
from unittest.mock import patch

from learning_architect import (
    AI_PROVIDER_LABELS,
    _review_resource,
    build_ai_messages,
    build_request_payload,
    build_responses_api_request,
    extract_sources,
    generate_roadmap,
    generate_roadmap_browse,
    generate_roadmap_offline,
    normalize_request,
)


class LearningArchitectTests(unittest.TestCase):
    @staticmethod
    def _minimal_roadmap() -> dict:
        resource = {
            "search_query": "official docs",
            "title": "official docs",
            "url": "",
            "source_type": "Official documentation",
            "access_note": "note",
            "why_this_resource": "why",
            "contribution_to_path": "how",
            "primary_focus": "focus",
            "time_fit": "fit",
            "use_strategy": "use",
        }
        return {
            "industry_insight": "Sentence one. Sentence two.",
            "weekly_breakdown": [
                {
                    "week": 1,
                    "focus": "Basics",
                    "why_this_week": "Reason.",
                    "priority_focus": "Most important thing.",
                    "time_budget": "5 hours.",
                    "execution_plan": "Do this then that.",
                    "resources": [resource, dict(resource), dict(resource)],
                    "hands_on_project": "Do thing",
                }
            ],
            "adjustment_log": "OpenAI test.",
        }

    def test_normalize_aliases_and_defaults(self) -> None:
        normalized, assumptions = normalize_request(
            {
                "subject": "Cybersecurity",
                "level": "intermediate",
                "hours per week": "5 hours",
            }
        )

        self.assertEqual(normalized["topic"], "Cybersecurity")
        self.assertEqual(normalized["experience_level"], "intermediate")
        self.assertEqual(normalized["time_available_per_week"], "5 hours")
        self.assertEqual(normalized["schedule_length"], "8 weeks")
        self.assertTrue(any("schedule_length" in item for item in assumptions))

    def test_infer_topic_from_job_title(self) -> None:
        normalized, assumptions = normalize_request({"target_job_title": "MLOps Engineer"})

        self.assertEqual(normalized["topic"], "MLOps Engineer")
        self.assertTrue(any("target_job_title" in item for item in assumptions))

    def test_build_payload_contains_strict_schema(self) -> None:
        payload = build_request_payload("Frontend Development")

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertIn("weekly_breakdown", payload["response_format"]["json_schema"]["schema"]["properties"])

    def test_build_responses_request_includes_web_search(self) -> None:
        request_body = build_responses_api_request("Data Engineering")

        self.assertEqual(request_body["model"], "gpt-5.2")
        self.assertEqual(request_body["text"]["format"]["type"], "json_schema")
        self.assertEqual(request_body["tools"][0]["type"], "web_search")
        self.assertEqual(request_body["tool_choice"], "auto")

    def test_extract_sources_reads_web_search_results(self) -> None:
        response_json = {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Example Source",
                                "url": "https://example.com/source",
                            }
                        ]
                    },
                }
            ]
        }

        self.assertEqual(
            extract_sources(response_json),
            [{"title": "Example Source", "url": "https://example.com/source"}],
        )

    def test_offline_generation_returns_exact_resource_count(self) -> None:
        result = generate_roadmap_offline(
            {
                "topic": "Data Engineering",
                "schedule_length": "6 weeks",
                "secondary_goal": "analytics engineering portfolio",
            }
        )

        roadmap = result["roadmap"]
        self.assertEqual(len(roadmap["weekly_breakdown"]), 6)
        self.assertIn("offline inference", roadmap["industry_insight"])
        self.assertTrue(all(len(week["resources"]) == 3 for week in roadmap["weekly_breakdown"]))
        self.assertTrue(all("why_this_week" in week for week in roadmap["weekly_breakdown"]))
        self.assertTrue(all("priority_focus" in week for week in roadmap["weekly_breakdown"]))
        self.assertIn("Resource review pass:", roadmap["adjustment_log"])
        self.assertTrue(
            all(
                resource["url"].startswith("https://")
                for week in roadmap["weekly_breakdown"]
                for resource in week["resources"]
            )
        )
        self.assertTrue(
            all("time_fit" in resource for week in roadmap["weekly_breakdown"] for resource in week["resources"])
        )

    @patch("learning_architect.discover_best_resources")
    def test_browse_generation_enriches_resources(self, mock_discover_best_resources) -> None:
        mock_discover_best_resources.side_effect = lambda queries, timeout=12, max_workers=6: {
            query: {
                "query": query,
                "title": f"Best match for {query}",
                "url": f"https://example.com/{index}",
                "display": f"Web: Best match for {query} | https://example.com/{index}",
                "source_label": "Official documentation",
                "live": True,
            }
            for index, query in enumerate(queries, start=1)
        }

        result = generate_roadmap_browse("Frontend Development")

        self.assertEqual(result["provider"], "browse")
        self.assertGreater(result["live_resource_hits"], 0)
        self.assertTrue(
            all(
                resource["url"].startswith("https://example.com/")
                for week in result["roadmap"]["weekly_breakdown"]
                for resource in week["resources"]
            )
        )
        self.assertTrue(
            all(
                resource["access_note"].startswith("Direct free resource found")
                for week in result["roadmap"]["weekly_breakdown"]
                for resource in week["resources"]
            )
        )

    def test_default_provider_is_browse(self) -> None:
        with patch("learning_architect.discover_best_resources") as mock_discover_best_resources:
            mock_discover_best_resources.return_value = {}
            result = generate_roadmap("Frontend Development")

        self.assertEqual(result["provider"], "browse")
        self.assertGreaterEqual(len(result["roadmap"]["weekly_breakdown"]), 1)

    def test_build_ai_messages_includes_system_and_user(self) -> None:
        payload = build_ai_messages({"topic": "Machine Learning"})

        self.assertIn("Return only valid JSON", payload["system"])
        self.assertIn("Topic: Machine Learning", payload["user"])

    def test_review_resource_upgrades_paid_or_weak_sources_to_trusted_free(self) -> None:
        resource = {
            "search_query": "Udemy backend bootcamp",
            "title": "Udemy backend bootcamp",
            "url": "https://www.udemy.com/course/backend-bootcamp",
            "source_type": "Free web resource",
            "access_note": "",
            "why_this_resource": "",
            "contribution_to_path": "",
            "primary_focus": "",
            "time_fit": "",
            "use_strategy": "",
        }

        reviewed = _review_resource(resource, resource_index=0, week_focus="Backend fundamentals", hours_per_week=6)

        self.assertEqual(reviewed["source_type"], "Official documentation")
        self.assertNotIn("udemy", reviewed["search_query"].lower())
        self.assertTrue(reviewed["url"].startswith("https://"))
        self.assertIn("Quality review:", reviewed["access_note"])

    def test_review_resource_shortens_oversized_resource_for_timeline(self) -> None:
        resource = {
            "search_query": "freeCodeCamp machine learning full course complete bootcamp",
            "title": "freeCodeCamp machine learning full course complete bootcamp",
            "url": "",
            "source_type": "freeCodeCamp",
            "access_note": "",
            "why_this_resource": "",
            "contribution_to_path": "",
            "primary_focus": "",
            "time_fit": "",
            "use_strategy": "",
        }

        reviewed = _review_resource(resource, resource_index=1, week_focus="Machine learning basics", hours_per_week=4)

        self.assertIn("focused module", reviewed["search_query"].lower())
        self.assertIn("minutes", reviewed["time_fit"].lower())
        self.assertIn("timeline cap", reviewed["access_note"].lower())

    @patch("learning_architect._http_json_request")
    def test_ollama_provider_parses_json_content(self, mock_http_json_request) -> None:
        mock_http_json_request.return_value = {
            "message": {
                "content": "{\"industry_insight\":\"Sentence one. Sentence two.\",\"weekly_breakdown\":[{\"week\":1,\"focus\":\"Basics\",\"why_this_week\":\"Reason.\",\"priority_focus\":\"Most important thing.\",\"time_budget\":\"5 hours.\",\"execution_plan\":\"Do this then that.\",\"resources\":[{\"search_query\":\"a\",\"title\":\"a\",\"url\":\"\",\"source_type\":\"Official documentation\",\"access_note\":\"Planned query.\",\"why_this_resource\":\"Why a.\",\"contribution_to_path\":\"How a helps.\",\"primary_focus\":\"Focus area.\",\"time_fit\":\"Fits the timeframe.\",\"use_strategy\":\"Use it this way.\"},{\"search_query\":\"b\",\"title\":\"b\",\"url\":\"\",\"source_type\":\"YouTube tutorial\",\"access_note\":\"Planned query.\",\"why_this_resource\":\"Why b.\",\"contribution_to_path\":\"How b helps.\",\"primary_focus\":\"Focus area.\",\"time_fit\":\"Fits the timeframe.\",\"use_strategy\":\"Use it this way.\"},{\"search_query\":\"c\",\"title\":\"c\",\"url\":\"\",\"source_type\":\"GitHub repository\",\"access_note\":\"Planned query.\",\"why_this_resource\":\"Why c.\",\"contribution_to_path\":\"How c helps.\",\"primary_focus\":\"Focus area.\",\"time_fit\":\"Fits the timeframe.\",\"use_strategy\":\"Use it this way.\"}],\"hands_on_project\":\"Do thing\"}],\"adjustment_log\":\"Local AI test.\"}"
            }
        }

        result = generate_roadmap("Frontend Development", provider="ollama")

        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["ai_provider_label"], AI_PROVIDER_LABELS["ollama"])
        self.assertEqual(result["roadmap"]["weekly_breakdown"][0]["week"], 1)
        self.assertIn("Resource review pass:", result["roadmap"]["adjustment_log"])
        self.assertIn(
            "minutes",
            result["roadmap"]["weekly_breakdown"][0]["resources"][0]["time_fit"].lower(),
        )

    @patch("learning_architect._http_json_request")
    def test_openai_provider_accepts_direct_api_key(self, mock_http_json_request) -> None:
        roadmap = self._minimal_roadmap()

        def _side_effect(url, body, headers, timeout=120):
            self.assertIn("Authorization", headers)
            self.assertIn("custom_openai_key", headers["Authorization"])
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(roadmap)}],
                    }
                ]
            }

        mock_http_json_request.side_effect = _side_effect

        result = generate_roadmap("Frontend Development", provider="openai", api_key="custom_openai_key")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["roadmap"]["weekly_breakdown"][0]["week"], 1)

    @patch("learning_architect._generate_with_gemini")
    def test_generate_roadmap_dispatches_to_gemini(self, mock_generate_with_gemini) -> None:
        mock_generate_with_gemini.return_value = {
            "roadmap": {
                "industry_insight": "Sentence one. Sentence two.",
                "weekly_breakdown": [
                    {
                        "week": 1,
                        "focus": "Basics",
                        "why_this_week": "Reason.",
                        "priority_focus": "Most important thing.",
                        "time_budget": "5 hours.",
                        "execution_plan": "Do this then that.",
                        "resources": [
                            {
                                "search_query": "a",
                                "title": "a",
                                "url": "",
                                "source_type": "Official documentation",
                                "access_note": "Planned query.",
                                "why_this_resource": "Why a.",
                                "contribution_to_path": "How a helps.",
                                "primary_focus": "Focus area.",
                                "time_fit": "Fits the timeframe.",
                                "use_strategy": "Use it this way.",
                            },
                            {
                                "search_query": "b",
                                "title": "b",
                                "url": "",
                                "source_type": "YouTube tutorial",
                                "access_note": "Planned query.",
                                "why_this_resource": "Why b.",
                                "contribution_to_path": "How b helps.",
                                "primary_focus": "Focus area.",
                                "time_fit": "Fits the timeframe.",
                                "use_strategy": "Use it this way.",
                            },
                            {
                                "search_query": "c",
                                "title": "c",
                                "url": "",
                                "source_type": "GitHub repository",
                                "access_note": "Planned query.",
                                "why_this_resource": "Why c.",
                                "contribution_to_path": "How c helps.",
                                "primary_focus": "Focus area.",
                                "time_fit": "Fits the timeframe.",
                                "use_strategy": "Use it this way.",
                            },
                        ],
                        "hands_on_project": "Do thing",
                    }
                ],
                "adjustment_log": "Gemini test.",
            },
            "provider": "gemini",
            "sources": [],
        }

        result = generate_roadmap("Data Engineering", provider="gemini")

        self.assertEqual(result["provider"], "gemini")
        mock_generate_with_gemini.assert_called_once()


if __name__ == "__main__":
    unittest.main()
