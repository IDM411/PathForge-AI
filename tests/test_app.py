import unittest
from unittest.mock import patch

from app import _default_form_state, main, render_page
from learning_architect import generate_roadmap_offline


class AppRenderTests(unittest.TestCase):
    def test_render_page_includes_progress_checklist_ui(self) -> None:
        result = generate_roadmap_offline("Data Engineering")
        page = render_page(_default_form_state(), result=result)

        self.assertIn("resource_progress_card", page)
        self.assertIn("resource_progress_percent", page)
        self.assertIn("resource-check", page)
        self.assertIn("Next up", page)
        self.assertIn("app-header", page)
        self.assertIn("open-wizard-trigger", page)
        self.assertIn("path_preview_line", page)
        self.assertIn("resource-summary", page)
        self.assertIn("week-summary", page)
        self.assertNotIn("How to approach it", page)
        self.assertNotIn("How it contributes", page)
        self.assertIn("page_loading_fill", page)
        self.assertIn("applyViewMode()", page)
        self.assertIn("settings_modal", page)
        self.assertIn("open-settings-trigger", page)
        self.assertIn("settings_openai_api_key", page)
        self.assertNotIn("button-secondary open-wizard-trigger", page)

    @patch("app.ThreadingHTTPServer")
    @patch("builtins.print")
    def test_main_handles_keyboard_interrupt_without_traceback(self, mock_print, mock_server_class) -> None:
        mock_server = mock_server_class.return_value
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        main()

        mock_server.server_close.assert_called_once()
        mock_print.assert_any_call("\nShutting down AI Learning Architect.")


if __name__ == "__main__":
    unittest.main()
