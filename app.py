import html
import hashlib
import json
import os
import re
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from learning_architect import AI_PROVIDER_LABELS, build_ai_messages, generate_roadmap


FORM_FIELDS = [
    ("topic", "Topic", "Data Engineering, UX Design, Cybersecurity, Product Analytics"),
    ("experience_level", "Experience level", "beginner, intermediate, advanced"),
    ("schedule_length", "Schedule length", "8 weeks, 12 weeks, 3 months"),
    ("time_available_per_week", "Time per week", "5 hours, 8-10 hours"),
    ("target_job_title", "Target job title", "Data Engineer, SOC Analyst, Frontend Developer"),
    ("job_industry_focus", "Job or industry focus", "healthcare, fintech, e-commerce, general career relevance"),
    ("domain_specialization", "Domain specialization", "analytics engineering, app security, MLOps"),
    ("secondary_goal", "Secondary goal", "portfolio building, interview prep, freelancing"),
]

PROVIDER_SETTINGS_FIELDS = [
    "openai_api_key",
    "gemini_api_key",
    "anthropic_api_key",
    "deepseek_api_key",
    "perplexity_api_key",
    "openai_compatible_api_key",
    "openai_compatible_base_url",
    "ollama_base_url",
]


def _pretty_json(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2))


def _escape_attr(value: str) -> str:
    return html.escape(value, quote=True)


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _default_form_state() -> Dict[str, str]:
    state = {field: "" for field, _, _ in FORM_FIELDS}
    state.update(
        {
            "custom_modifications": "",
            "raw_input": "",
            "mode": "browse",
            "input_method": "guided",
            "view_mode": "grid",
        }
    )
    return state


def _merge_form_state(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    state = _default_form_state()
    if overrides:
        state.update({key: value for key, value in overrides.items() if value is not None})
    return state


def _parse_raw_input(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Raw input cannot be empty.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _build_user_input_from_state(form_state: Dict[str, str]) -> Any:
    if form_state.get("input_method") == "raw":
        return _parse_raw_input(form_state.get("raw_input", ""))

    payload: Dict[str, str] = {}
    for field, _, _ in FORM_FIELDS:
        value = form_state.get(field, "").strip()
        if value:
            payload[field] = value
    custom_modifications = form_state.get("custom_modifications", "").strip()
    if custom_modifications:
        payload["custom_modifications"] = custom_modifications
    if not payload:
        raise ValueError("Add at least a topic or switch to raw input mode.")
    return payload


def _resource_domain(url: str) -> str:
    if not url:
        return ""
    cleaned = url.replace("https://", "").replace("http://", "")
    return cleaned.split("/", 1)[0]


def _roadmap_progress_key(roadmap: Dict[str, Any]) -> str:
    payload = json.dumps(roadmap, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _trim_words(value: str, limit: int = 8) -> str:
    words = value.split()
    if len(words) <= limit:
        return value
    return " ".join(words[:limit]) + "..."


def _trim_sentences(value: str, max_sentences: int = 1) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if not compact:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", compact)
    trimmed = " ".join(parts[:max_sentences]).strip()
    if not re.search(r"[.!?]$", trimmed):
        trimmed += "."
    return trimmed


def _render_path_preview(roadmap: Dict[str, Any]) -> str:
    steps: List[str] = []
    for week in roadmap.get("weekly_breakdown", []):
        week_number = int(week.get("week", 0))
        focus = _trim_words(str(week.get("focus", "")), limit=8)
        steps.append(
            f"""
            <li class="path-step" data-path-week="{week_number}">
              <span class="path-dot" aria-hidden="true"></span>
              <div class="path-step-copy">
                <strong>Week {week_number}</strong>
                <span>{html.escape(focus)}</span>
              </div>
            </li>
            """
        )
    return "".join(steps)


def _render_week_cards(
    roadmap: Dict[str, Any],
    resource_details_by_week: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    cards: List[str] = []
    for week in roadmap["weekly_breakdown"]:
        details = resource_details_by_week.get(str(week["week"]), []) if resource_details_by_week else []
        resource_items: List[str] = []
        for index, resource in enumerate(week["resources"]):
            detail = details[index] if index < len(details) else {}
            url = str(resource.get("url", ""))
            item_key = f"week-{week['week']}-resource-{index + 1}"
            source_type = html.escape(str(resource.get("source_type") or detail.get("source_label") or "Resource"))
            status = "Live" if detail.get("live") else ("Direct" if url else "Planned")
            badge_class = "badge-live" if detail.get("live") else "badge-fallback"
            domain = html.escape(_resource_domain(url))

            if detail.get("live") and url:
                title_html = (
                    f"<a class='resource-link' href='{html.escape(url)}' "
                    f"target='_blank' rel='noreferrer'>{html.escape(str(resource.get('title') or resource.get('search_query') or 'Resource'))}</a>"
                )
                action_html = (
                    f"<a class='resource-action primary' href='{html.escape(url)}' "
                    f"target='_blank' rel='noreferrer'>Open resource</a>"
                )
            else:
                title_html = f"<span class='resource-title'>{html.escape(str(resource.get('title') or resource.get('search_query') or 'Resource'))}</span>"
                action_html = (
                    f"<a class='resource-action secondary' href='{html.escape(url)}' "
                    f"target='_blank' rel='noreferrer'>Open topic link</a>"
                    if url
                    else "<span class='resource-action disabled'>Direct link unavailable</span>"
                )

            domain_html = f"<span class='resource-domain'>{domain}</span>" if domain else ""
            resource_summary = _trim_sentences(
                f"{resource.get('why_this_resource', '')} {resource.get('use_strategy', '')}",
                max_sentences=2,
            )
            resource_items.append(
                f"""
                <li class="resource-item" data-progress-item="{item_key}" data-week="{week['week']}" data-resource-number="{index + 1}">
                  <div class="resource-check-row">
                    <label class="resource-check-label">
                      <input class="resource-check" type="checkbox" data-progress-item="{item_key}" data-week="{week['week']}" data-resource-number="{index + 1}">
                      <span>Mark this resource complete</span>
                    </label>
                    <span class="resource-next-badge">Next up</span>
                  </div>
                  <div class="resource-meta">
                    <span class="badge {badge_class}">{status}</span>
                    <span class="source-label">{source_type}</span>
                    {domain_html}
                  </div>
                  <div class="resource-main">{title_html}</div>
                  <div class="resource-actions">{action_html}</div>
                  <p class="resource-summary">{html.escape(resource_summary)}</p>
                </li>
                """
            )

        week_summary = _trim_sentences(
            f"{week.get('why_this_week', '')} {week.get('priority_focus', '')} {week.get('execution_plan', '')}",
            max_sentences=1,
        )
        cards.append(
            f"""
            <article class="week-card">
              <div class="week-top">
                <span class="week-number">Week {week["week"]}</span>
                <span class="week-progress-pill" data-week-progress="{week['week']}">0% complete</span>
              </div>
              <div class="week-body">
                <h3>{html.escape(week["focus"])}</h3>
                <p class="week-summary">{html.escape(week_summary)}</p>
                <p class="project-label">Hands-on project</p>
                <p class="project-text">{html.escape(week["hands_on_project"])}</p>
                <p class="resource-label">Best free resources found</p>
                <ul class="resource-list">
                  {''.join(resource_items)}
                </ul>
              </div>
            </article>
            """
        )
    return "".join(cards)


def _render_guided_fields(form_state: Dict[str, str]) -> str:
    return ""


def _render_guided_hidden_inputs(form_state: Dict[str, str]) -> str:
    inputs: List[str] = []
    for field, _, _ in FORM_FIELDS:
        inputs.append(
            f"<input type='hidden' name='{field}' id='hidden_{field}' value='{_escape_attr(form_state.get(field, ''))}'>"
        )
    inputs.append(
        f"<input type='hidden' name='custom_modifications' id='hidden_custom_modifications' value='{_escape_attr(form_state.get('custom_modifications', ''))}'>"
    )
    return "".join(inputs)


def _render_provider_settings_hidden_inputs() -> str:
    return "".join(
        f"<input type='hidden' name='{field}' id='hidden_{field}' value=''>"
        for field in PROVIDER_SETTINGS_FIELDS
    )


def _render_guided_summary(form_state: Dict[str, str]) -> str:
    labels = {field: label for field, label, _ in FORM_FIELDS}
    labels["custom_modifications"] = "Custom modifications"
    summary_items: List[str] = []
    for key in [field for field, _, _ in FORM_FIELDS] + ["custom_modifications"]:
        value = form_state.get(key, "").strip()
        if not value:
            continue
        summary_items.append(
            f"""
            <div class="summary-chip">
              <span class="summary-chip-label">{html.escape(labels[key])}</span>
              <strong>{html.escape(value)}</strong>
            </div>
            """
        )
    if not summary_items:
        return "<p class='wizard-empty'>No guided answers yet. Start a new path and answer only what you know.</p>"
    return "".join(summary_items)


def _render_wizard_steps(form_state: Dict[str, str]) -> str:
    question_fields = FORM_FIELDS + [
        (
            "custom_modifications",
            "Custom modifications",
            "Anything specific: certification focus, no video resources, project-heavy, beginner-friendly pacing",
        )
    ]
    steps: List[str] = []
    total = len(question_fields)
    for index, (field, label, placeholder) in enumerate(question_fields, start=1):
        is_textarea = field == "custom_modifications"
        value = html.escape(form_state.get(field, ""))
        control = (
            f"<textarea id='wizard_{field}' class='wizard-control' data-field='{field}' placeholder='{html.escape(placeholder)}'>{value}</textarea>"
            if is_textarea
            else f"<input id='wizard_{field}' class='wizard-control' data-field='{field}' type='text' value='{value}' placeholder='{html.escape(placeholder)}'>"
        )
        requirement = "<span class='wizard-required'>Required</span>" if field == "topic" else "<span class='wizard-optional'>Optional</span>"
        steps.append(
            f"""
            <section class="wizard-step" data-step="{index}" data-field="{field}">
              <div class="wizard-progress-row">
                <span class="wizard-progress-kicker">Question {index} of {total}</span>
                {requirement}
              </div>
              <h3>{html.escape(label)}</h3>
              <p>{html.escape(placeholder)}</p>
              {control}
            </section>
            """
        )
    return "".join(steps)


def _api_key_field_for_mode(mode: str) -> Optional[str]:
    return {
        "openai": "openai_api_key",
        "gemini": "gemini_api_key",
        "anthropic": "anthropic_api_key",
        "deepseek": "deepseek_api_key",
        "perplexity": "perplexity_api_key",
        "openai_compatible": "openai_compatible_api_key",
    }.get(mode)


def _env_overrides_from_form_state(form_state: Dict[str, str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    openai_compatible_base_url = form_state.get("openai_compatible_base_url", "").strip()
    ollama_base_url = form_state.get("ollama_base_url", "").strip()
    if openai_compatible_base_url:
        overrides["OPENAI_COMPATIBLE_BASE_URL"] = openai_compatible_base_url
    if ollama_base_url:
        overrides["OLLAMA_BASE_URL"] = ollama_base_url
    return overrides


def render_page(
    form_state: Dict[str, str],
    result: Optional[Dict[str, Any]] = None,
    preview_json: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> str:
    mode = form_state.get("mode", "browse")
    view_mode = form_state.get("view_mode", "grid")
    grid_checked = "checked" if view_mode == "grid" else ""
    list_checked = "checked" if view_mode == "list" else ""
    weeks_class = "weeks-grid list-view" if view_mode == "list" else "weeks-grid"
    selected = lambda value: "selected" if mode == value else ""
    wizard_button_label = "Edit path answers" if _render_guided_summary(form_state).find("wizard-empty") == -1 else "Start new path"
    wizard_summary_html = _render_guided_summary(form_state)
    hidden_inputs_html = _render_guided_hidden_inputs(form_state)
    provider_settings_hidden_inputs_html = _render_provider_settings_hidden_inputs()
    wizard_steps_html = _render_wizard_steps(form_state)

    result_section = """
    <section class="summary-card empty-state">
      <span class="section-kicker">Start here</span>
      <h2>Start with a topic.</h2>
      <p>Answer only what you know and the planner fills the rest.</p>
      <div class="empty-state-grid">
        <div class="empty-state-panel">
          <span class="guidance-label">Guided</span>
          <p>Use <strong>Start new path</strong> for a quick question-by-question setup.</p>
        </div>
        <div class="empty-state-panel">
          <span class="guidance-label">Quick start</span>
          <p>Choose a mode and generate right away.</p>
        </div>
      </div>
      <div class="example-wrap">
        <span class="section-kicker subdued">Example prompts</span>
        <div class="example-grid">
          <div class="example-chip">Data Engineering for 10 weeks</div>
          <div class="example-chip">Beginner Cybersecurity for SOC Analyst roles</div>
          <div class="example-chip">Product Analytics with a portfolio focus</div>
          <div class="example-chip">Frontend Development for ecommerce jobs</div>
        </div>
      </div>
    </section>
    """

    if error:
        result_section = f"<div class='error'>{html.escape(error)}</div>"
    elif result is not None:
        roadmap = result["roadmap"]
        progress_key = _roadmap_progress_key(roadmap)
        total_resources = sum(len(week.get("resources", [])) for week in roadmap.get("weekly_breakdown", []))
        live_hits = result.get("live_resource_hits", 0)
        fallback_hits = result.get("fallback_resource_hits", 0)
        weeks = len(roadmap["weekly_breakdown"])
        provider = result.get("provider", "browse")
        provider_name = {"browse": "Live Browse", "offline": "Offline", **AI_PROVIDER_LABELS}.get(provider, provider.title())
        provider_copy = {
            "browse": "Public web discovery with no AI required.",
            "offline": "Local planning only, with no network dependency.",
            "ollama": "Local open-source model on your machine.",
            "openai": "Hosted OpenAI model using your own API key.",
            "openai_compatible": "Any OpenAI-compatible endpoint or local server.",
            "gemini": "Hosted Gemini model using your own API key.",
            "anthropic": "Hosted Claude model using your own API key.",
            "deepseek": "Hosted DeepSeek model using your own API key.",
            "perplexity": "Hosted Perplexity model using your own API key.",
        }.get(provider, "Configured AI provider.")
        result_section = f"""
        <section class="summary-card path-preview-card">
          <div class="path-preview-head">
            <h2>Path Preview</h2>
            <p>Track what comes next and see this path light up as you complete resources.</p>
          </div>
          <ol class="path-preview-line" id="path_preview_line">
            {_render_path_preview(roadmap)}
          </ol>
        </section>
        <section class="hero-stats">
          <div class="stat-card">
            <span class="stat-kicker">Provider</span>
            <strong>{provider_name}</strong>
            <p>{provider_copy}</p>
          </div>
          <div class="stat-card">
            <span class="stat-kicker">Weeks</span>
            <strong>{weeks}</strong>
            <p>Goal-oriented sequence matched to your timeframe.</p>
          </div>
          <div class="stat-card">
            <span class="stat-kicker">Direct Links</span>
            <strong>{live_hits}</strong>
            <p>Resources with an actual destination page.</p>
          </div>
          <div class="stat-card">
            <span class="stat-kicker">Direct Fallbacks</span>
            <strong>{fallback_hits}</strong>
            <p>These could not be matched live and were routed to direct source sites.</p>
          </div>
        </section>
        <section class="summary-card">
          <h2>Industry Insight</h2>
          <p>{html.escape(roadmap["industry_insight"])}</p>
          <h2>Adjustment Log</h2>
          <p>{html.escape(roadmap["adjustment_log"])}</p>
        </section>
        <section class="summary-card progress-card" id="resource_progress_card" data-progress-key="{progress_key}">
          <div class="progress-card-top">
            <div>
              <span class="stat-kicker">Learning Progress</span>
              <h2>Check off resources as you finish them</h2>
              <p id="resource_progress_copy">0 of {total_resources} resources completed.</p>
            </div>
            <div class="progress-score">
              <strong id="resource_progress_percent">0%</strong>
              <span>completed</span>
            </div>
          </div>
          <div class="completion-bar" aria-hidden="true">
            <div class="completion-fill" id="resource_progress_fill"></div>
          </div>
          <div class="progress-meta-row">
            <span class="progress-meta-pill">{total_resources} total resource checkpoints</span>
            <span class="progress-meta-pill next-step" id="resource_next_step">Next up: Week 1, Resource 1</span>
          </div>
        </section>
        <section class="{weeks_class}">
          {_render_week_cards(roadmap, result.get("resource_details_by_week"))}
        </section>
        <details class="json-panel">
          <summary>Raw JSON output</summary>
          <pre>{_pretty_json(roadmap)}</pre>
        </details>
        """
    elif preview_json is not None:
        result_section = f"""
        <section class="summary-card">
          <h2>Request Preview</h2>
          <p>This shows the normalized request or AI prompt bundle before generation starts.</p>
        </section>
        <pre>{_pretty_json(preview_json)}</pre>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Learning Architect</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #121423;
      --bg-soft: #1a1e30;
      --panel: rgba(36, 31, 56, 0.78);
      --panel-strong: rgba(40, 34, 64, 0.92);
      --panel-lite: rgba(61, 52, 93, 0.72);
      --ink: #f4f0ff;
      --muted: #c7bedf;
      --accent: #8b5cf6;
      --accent-deep: #6d28d9;
      --accent-hot: #22d3ee;
      --line: rgba(180, 152, 255, 0.22);
      --shadow: rgba(5, 2, 16, 0.45);
      --live-bg: rgba(34, 211, 238, 0.14);
      --live-ink: #8ff2ff;
      --fallback-bg: rgba(251, 191, 36, 0.16);
      --fallback-ink: #f6d365;
      --error-bg: rgba(127, 29, 29, 0.6);
      --error-ink: #ffd6d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Manrope", "Segoe UI", sans-serif;
      text-align: center;
      background:
        radial-gradient(circle at 12% -8%, rgba(139, 92, 246, 0.18), transparent 30%),
        radial-gradient(circle at 88% 0%, rgba(125, 94, 194, 0.12), transparent 28%),
        linear-gradient(180deg, #0e1324 0%, var(--bg) 42%, var(--bg-soft) 100%);
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(188, 170, 228, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(188, 170, 228, 0.02) 1px, transparent 1px);
      background-size: 64px 64px;
      opacity: 0.22;
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 18px 48px;
      position: relative;
      z-index: 1;
    }}
    .app-shell {{
      display: grid;
      gap: 20px;
    }}
    .app-header {{
      display: flex;
      align-items: end;
      justify-content: center;
      gap: 18px;
      flex-wrap: wrap;
      padding: 24px 26px;
      background: linear-gradient(135deg, rgba(36, 24, 66, 0.92), rgba(18, 13, 31, 0.96));
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: 0 18px 50px var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .app-header-copy {{
      max-width: 760px;
    }}
    .app-header-actions {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 440px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .sidebar {{
      position: sticky;
      top: 18px;
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 50px var(--shadow);
      padding: 24px;
      backdrop-filter: blur(14px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      font-size: clamp(2.2rem, 5vw, 3.9rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin: 0 0 10px;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      font-size: 1.15rem;
    }}
    h3 {{
      margin: 0 0 12px;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      font-size: 1.12rem;
      line-height: 1.25;
    }}
    p {{
      margin: 0 0 12px;
      line-height: 1.6;
      color: var(--muted);
    }}
    .kicker {{
      display: inline-block;
      margin-bottom: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(139, 92, 246, 0.14);
      color: #e3d7ff;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .section-kicker {{
      display: inline-block;
      margin-bottom: 10px;
      color: #d9caff;
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .section-kicker.subdued {{
      color: #b9abd8;
      margin-bottom: 8px;
    }}
    .hero-note {{
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.16), rgba(34, 211, 238, 0.12));
      border: 1px solid rgba(139, 92, 246, 0.18);
      border-radius: 16px;
      padding: 14px 16px;
      color: #efe7ff;
      font-weight: 600;
    }}
    .hero-note.compact {{
      padding: 12px 14px;
      font-size: 0.94rem;
    }}
    .control-panel label {{
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
      color: var(--ink);
      text-align: center;
    }}
    input, textarea, select, button {{
      width: 100%;
      font: inherit;
    }}
    input, textarea, select {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-strong);
      padding: 12px 14px;
      color: var(--ink);
      text-align: center;
    }}
    input::placeholder, textarea::placeholder {{
      color: #9a8dbb;
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    .mode-row {{
      display: grid;
      gap: 14px;
    }}
    .mode-note {{
      margin: 8px 0 0;
      font-size: 0.9rem;
      color: #cfc2ea;
    }}
    .input-switch {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin: 6px 0 2px;
    }}
    .input-switch label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      margin: 0;
    }}
    .input-switch input {{
      width: auto;
      margin: 0;
    }}
    .guided-panel {{
      display: none;
      gap: 12px;
    }}
    .guided-panel.active {{
      display: grid;
    }}
    .wizard-launcher {{
      display: grid;
      gap: 14px;
    }}
    .wizard-summary {{
      display: grid;
      gap: 10px;
    }}
    .wizard-empty {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px dashed rgba(180, 152, 255, 0.2);
    }}
    .summary-chip {{
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(180, 152, 255, 0.12);
    }}
    .summary-chip-label {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.76rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #cbbdff;
      font-weight: 800;
    }}
    .summary-chip strong {{
      font-size: 0.98rem;
      color: var(--ink);
    }}
    .form-section {{
      border: 1px solid rgba(180, 152, 255, 0.12);
      background: rgba(255, 255, 255, 0.02);
      border-radius: 18px;
      padding: 16px;
    }}
    .section-head {{
      margin-bottom: 12px;
    }}
    .section-head h3 {{
      margin-bottom: 4px;
    }}
    .section-head p {{
      margin-bottom: 0;
      font-size: 0.95rem;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .field-wide {{
      margin-top: 2px;
    }}
    .raw-panel {{
      display: none;
    }}
    .raw-panel.active {{
      display: block;
    }}
    .raw-panel textarea {{
      min-height: 260px;
    }}
    .control-stack {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }}
    .provider-setup-row {{
      margin-top: 12px;
      padding: 12px;
      border: 1px solid rgba(180, 152, 255, 0.18);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .provider-setup-row p {{
      margin: 0;
      font-size: 0.92rem;
      color: #d8cdee;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 16px 18px;
      font-weight: 800;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent-hot));
      box-shadow: 0 14px 30px rgba(139, 92, 246, 0.28);
      cursor: pointer;
    }}
    .button-secondary {{
      width: auto;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(180, 152, 255, 0.16);
      box-shadow: none;
      color: var(--ink);
    }}
    .button-primary-inline {{
      width: auto;
      min-width: 180px;
    }}
    .button-secondary-inline {{
      width: auto;
      min-width: 150px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(180, 152, 255, 0.16);
      box-shadow: none;
      color: var(--ink);
      padding: 12px 16px;
    }}
    .settings-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }}
    .settings-field {{
      display: grid;
      gap: 6px;
    }}
    .settings-field.wide {{
      grid-column: 1 / -1;
    }}
    .settings-field label {{
      margin: 0;
      font-size: 0.84rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: #d8cbf2;
      font-weight: 700;
    }}
    .settings-help {{
      margin: 12px 0 0;
      color: #cabfe1;
      font-size: 0.9rem;
      line-height: 1.5;
    }}
    .settings-status {{
      min-height: 18px;
      color: #aef6ff;
      font-weight: 700;
      margin-top: 8px;
    }}
    .loading-shell {{
      position: fixed;
      inset: 0 0 auto 0;
      z-index: 80;
      opacity: 0;
      pointer-events: none;
      transition: opacity 180ms ease;
    }}
    .loading-shell.active {{
      opacity: 1;
    }}
    .loading-bar-track {{
      height: 6px;
      background: rgba(255,255,255,0.08);
      border-bottom: 1px solid rgba(180, 152, 255, 0.18);
      overflow: hidden;
    }}
    .loading-bar-fill {{
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-hot));
      box-shadow: 0 0 18px rgba(139, 92, 246, 0.35);
      transition: width 160ms ease;
    }}
    .loading-copy {{
      padding: 8px 14px;
      background: rgba(18, 14, 36, 0.94);
      color: #e6dcff;
      border-bottom: 1px solid rgba(180, 152, 255, 0.16);
      font-size: 0.82rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      font-weight: 800;
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 28px var(--shadow);
    }}
    .stat-kicker {{
      display: block;
      margin-bottom: 8px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #d7c5ff;
      font-weight: 700;
    }}
    .stat-card strong {{
      display: block;
      font-size: 1.8rem;
      line-height: 1;
      margin-bottom: 8px;
      color: var(--ink);
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 14px 36px var(--shadow);
      margin-bottom: 18px;
    }}
    .empty-state {{
      min-height: 220px;
    }}
    .empty-state-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .empty-state-panel {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(180, 152, 255, 0.12);
    }}
    .example-wrap {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid rgba(180, 152, 255, 0.12);
    }}
    .progress-card {{
      display: grid;
      gap: 14px;
    }}
    .path-preview-card {{
      display: grid;
      gap: 12px;
    }}
    .path-preview-head p {{
      margin-bottom: 0;
    }}
    .path-preview-line {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      position: relative;
    }}
    .path-step {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(180, 152, 255, 0.14);
      background: rgba(255, 255, 255, 0.03);
      transition: border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
    }}
    .path-dot {{
      width: 10px;
      height: 10px;
      margin-top: 6px;
      border-radius: 999px;
      border: 1px solid rgba(180, 152, 255, 0.28);
      background: rgba(255, 255, 255, 0.18);
      flex: 0 0 auto;
      transition: background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }}
    .path-step-copy {{
      display: grid;
      gap: 2px;
    }}
    .path-step-copy strong {{
      font-size: 0.84rem;
      color: #ece3ff;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .path-step-copy span {{
      font-size: 0.83rem;
      color: #c7bddf;
      line-height: 1.4;
    }}
    .path-step.active {{
      border-color: rgba(34, 211, 238, 0.36);
      background: rgba(34, 211, 238, 0.09);
    }}
    .path-step.active .path-dot {{
      background: #22d3ee;
      box-shadow: 0 0 16px rgba(34, 211, 238, 0.38);
      transform: scale(1.08);
    }}
    .path-step.complete {{
      border-color: rgba(139, 92, 246, 0.34);
      background: rgba(139, 92, 246, 0.14);
    }}
    .path-step.complete .path-dot {{
      background: #8b5cf6;
      box-shadow: 0 0 16px rgba(139, 92, 246, 0.36);
      transform: scale(1.08);
    }}
    .progress-card-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .progress-score {{
      min-width: 120px;
      text-align: center;
    }}
    .progress-score strong {{
      display: block;
      font-size: 2.2rem;
      line-height: 1;
      color: var(--ink);
    }}
    .progress-score span {{
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.78rem;
    }}
    .completion-bar {{
      height: 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(180, 152, 255, 0.14);
      overflow: hidden;
      position: relative;
    }}
    .completion-fill {{
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-hot));
      box-shadow: 0 0 28px rgba(139, 92, 246, 0.34);
      transition: width 180ms ease;
    }}
    .progress-meta-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .progress-meta-pill {{
      display: inline-flex;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(180, 152, 255, 0.12);
      color: #ece2ff;
      font-weight: 700;
      font-size: 0.88rem;
    }}
    .progress-meta-pill.next-step {{
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.14), rgba(34, 211, 238, 0.08));
    }}
    .example-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}
    .example-chip {{
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(180, 152, 255, 0.14);
      color: #ddd3f3;
      font-size: 0.9rem;
      font-weight: 600;
    }}
    .weeks-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .weeks-grid.list-view {{
      grid-template-columns: 1fr;
    }}
    .week-card {{
      background: linear-gradient(180deg, rgba(36, 24, 66, 0.92), rgba(24, 17, 45, 0.94));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 16px 34px var(--shadow);
    }}
    .week-body {{
      display: block;
    }}
    .week-top {{
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .week-number {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(139, 92, 246, 0.18);
      color: #eadcff;
      font-size: 0.82rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .week-progress-pill {{
      display: inline-flex;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(34, 211, 238, 0.12);
      color: #aef6ff;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .project-label, .resource-label {{
      margin: 14px 0 6px;
      color: var(--ink);
      font-size: 0.84rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .project-text {{
      color: var(--ink);
      font-weight: 500;
    }}
    .week-summary {{
      margin: 2px 0 8px;
      color: #dfd4f5;
      font-size: 0.95rem;
      line-height: 1.55;
    }}
    .resource-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    .resource-list li {{
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(180, 152, 255, 0.12);
      transition: border-color 180ms ease, background 180ms ease, opacity 180ms ease, transform 180ms ease;
    }}
    .resource-item.is-complete {{
      background: rgba(34, 211, 238, 0.08);
      border-color: rgba(34, 211, 238, 0.22);
      opacity: 0.86;
    }}
    .resource-item.is-next {{
      border-color: rgba(139, 92, 246, 0.42);
      box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.18);
      transform: translateY(-1px);
    }}
    .resource-check-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .resource-check-label {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin: 0;
      color: var(--ink);
      font-weight: 700;
    }}
    .resource-check-label input {{
      width: 18px;
      height: 18px;
      margin: 0;
      accent-color: #22d3ee;
      cursor: pointer;
      flex: 0 0 auto;
    }}
    .resource-next-badge {{
      display: none;
      width: fit-content;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(139, 92, 246, 0.14);
      color: #e8dcff;
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .resource-item.is-next .resource-next-badge {{
      display: inline-flex;
    }}
    .resource-meta {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .source-label, .resource-domain {{
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-weight: 700;
    }}
    .resource-link, .resource-title {{
      color: #f8f5ff;
      font-weight: 700;
      text-decoration: none;
      line-height: 1.45;
    }}
    .resource-link:hover {{
      text-decoration: underline;
    }}
    .resource-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
    }}
    .resource-summary {{
      margin: 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(180, 152, 255, 0.12);
      color: #e6dbfa;
      font-size: 0.92rem;
      line-height: 1.5;
    }}
    .guidance-label {{
      display: block;
      margin-bottom: 6px;
      font-size: 0.76rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #d9caff;
      font-weight: 800;
    }}
    .resource-action {{
      display: inline-flex;
      width: fit-content;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 800;
      text-decoration: none;
    }}
    .resource-action.primary {{
      background: rgba(139, 92, 246, 0.2);
      color: #f0eaff;
      border: 1px solid rgba(139, 92, 246, 0.25);
    }}
    .resource-action.secondary {{
      background: rgba(251, 191, 36, 0.12);
      color: #f6d365;
      border: 1px solid rgba(251, 191, 36, 0.2);
    }}
    .resource-action.disabled {{
      color: #8f86ad;
    }}
    .modal-shell {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(7, 6, 20, 0.72);
      backdrop-filter: blur(12px);
      z-index: 30;
    }}
    .modal-shell.active {{
      display: flex;
    }}
    .modal-card {{
      width: min(760px, 100%);
      max-height: min(88vh, 860px);
      overflow: auto;
      background: linear-gradient(180deg, rgba(29, 19, 56, 0.98), rgba(18, 13, 31, 0.98));
      border: 1px solid rgba(180, 152, 255, 0.2);
      border-radius: 26px;
      box-shadow: 0 30px 80px rgba(5, 2, 16, 0.58);
      padding: 24px;
    }}
    .modal-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .modal-head p {{
      margin-bottom: 0;
    }}
    .wizard-progress-panel {{
      display: grid;
      gap: 10px;
      margin: 2px 0 18px;
    }}
    .wizard-progress-bar {{
      position: relative;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(180, 152, 255, 0.12);
    }}
    .wizard-progress-fill {{
      width: 0%;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-hot));
      box-shadow: 0 0 24px rgba(139, 92, 246, 0.38);
      transition: width 180ms ease;
    }}
    .wizard-progress-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .wizard-progress-copy {{
      color: #e8dcff;
      font-weight: 700;
    }}
    .wizard-dots {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .wizard-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.14);
      border: 1px solid rgba(180, 152, 255, 0.18);
      transition: transform 180ms ease, background 180ms ease, box-shadow 180ms ease;
    }}
    .wizard-dot.active {{
      background: linear-gradient(180deg, var(--accent-hot), var(--accent));
      box-shadow: 0 0 14px rgba(34, 211, 238, 0.3);
      transform: scale(1.15);
    }}
    .wizard-dot.complete {{
      background: rgba(139, 92, 246, 0.56);
    }}
    .modal-close {{
      width: auto;
      min-width: 44px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(180, 152, 255, 0.16);
      box-shadow: none;
    }}
    .wizard-steps {{
      display: grid;
    }}
    .wizard-step {{
      display: none;
      gap: 12px;
    }}
    .wizard-step.active {{
      display: grid;
    }}
    .wizard-progress-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .wizard-progress-kicker {{
      font-size: 0.8rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #cbbdff;
      font-weight: 800;
    }}
    .wizard-required, .wizard-optional {{
      display: inline-flex;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .wizard-required {{
      background: rgba(34, 211, 238, 0.14);
      color: #8ff2ff;
    }}
    .wizard-optional {{
      background: rgba(255,255,255,0.06);
      color: #d5cfff;
    }}
    .wizard-control {{
      min-height: 58px;
      font-size: 1.02rem;
    }}
    textarea.wizard-control {{
      min-height: 160px;
    }}
    .wizard-footer {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    .wizard-footer button {{
      width: auto;
      min-width: 140px;
    }}
    .wizard-secondary {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(180, 152, 255, 0.16);
      box-shadow: none;
    }}
    .wizard-status {{
      min-height: 20px;
      color: #f6d365;
      font-weight: 700;
      margin-top: 10px;
    }}
    .badge {{
      display: inline-flex;
      width: fit-content;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge-live {{
      background: var(--live-bg);
      color: var(--live-ink);
    }}
    .badge-fallback {{
      background: var(--fallback-bg);
      color: var(--fallback-ink);
    }}
    .json-panel {{
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 26px var(--shadow);
    }}
    .json-panel summary {{
      cursor: pointer;
      font-weight: 800;
      color: var(--ink);
    }}
    pre {{
      margin: 14px 0 0;
      padding: 16px;
      border-radius: 16px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(7, 6, 20, 0.88);
      color: #f2f7f5;
    }}
    .error {{
      background: var(--error-bg);
      color: var(--error-ink);
      border: 1px solid rgba(248, 113, 113, 0.4);
      border-radius: 18px;
      padding: 16px;
      font-weight: 700;
    }}
    @media (max-width: 1120px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
      .app-header {{
        align-items: start;
      }}
    }}
    @media (max-width: 720px) {{
      .field-grid {{
        grid-template-columns: 1fr;
      }}
      .empty-state-grid {{
        grid-template-columns: 1fr;
      }}
      .progress-score {{
        text-align: center;
      }}
      .path-preview-line {{
        grid-template-columns: 1fr;
      }}
      .app-header-actions {{
        width: 100%;
        justify-content: center;
      }}
      .button-secondary,
      .button-primary-inline,
      .button-secondary-inline {{
        width: 100%;
      }}
      .settings-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (min-width: 860px) {{
      .weeks-grid.list-view .week-card {{
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        gap: 18px;
        align-items: start;
      }}
      .weeks-grid.list-view .week-top {{
        margin-bottom: 0;
        padding-top: 2px;
      }}
      .weeks-grid.list-view .week-body {{
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
        gap: 18px;
        align-items: start;
      }}
      .weeks-grid.list-view .week-body h3 {{
        grid-column: 1 / -1;
        margin-bottom: 2px;
      }}
      .weeks-grid.list-view .project-label,
      .weeks-grid.list-view .resource-label {{
        margin-top: 6px;
      }}
      .weeks-grid.list-view .resource-list li {{
        padding: 10px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="app-shell">
      <header class="app-header">
        <div class="app-header-copy">
          <span class="kicker">AI Learning Architect</span>
          <h1>Build a learning path in minutes.</h1>
          <p>Pick a mode, answer a few prompts, and get a practical weekly roadmap.</p>
        </div>
        <div class="app-header-actions">
          <button type="button" class="button-primary-inline open-wizard-trigger">{wizard_button_label}</button>
          <button type="button" class="button-secondary-inline open-settings-trigger">AI settings</button>
          <div class="hero-note compact">Browse and offline need no API key. AI modes use your own keys.</div>
        </div>
      </header>
    <section class="layout">
      <div class="sidebar">
        <div class="panel control-panel">
          <form method="post" id="planner_form">
            {provider_settings_hidden_inputs_html}
            <input type="hidden" name="input_method" value="guided">
            <div class="section-head">
              <span class="section-kicker">Planner controls</span>
              <h3>Roadmap setup</h3>
              <p>Pick a generation mode and layout, then use the guided wizard answers above.</p>
            </div>
            <div class="mode-row">
              <div>
                <label for="mode">Generation mode</label>
                <select id="mode" name="mode">
                  <optgroup label="No AI needed">
                    <option value="browse" {selected("browse")}>Browse live resources (recommended)</option>
                    <option value="offline" {selected("offline")}>Offline-only fallback</option>
                  </optgroup>
                  <optgroup label="Bring your own AI">
                    <option value="ollama" {selected("ollama")}>Local Ollama</option>
                    <option value="openai" {selected("openai")}>ChatGPT / OpenAI API</option>
                    <option value="gemini" {selected("gemini")}>Gemini API</option>
                    <option value="anthropic" {selected("anthropic")}>Claude API</option>
                    <option value="deepseek" {selected("deepseek")}>DeepSeek API</option>
                    <option value="perplexity" {selected("perplexity")}>Perplexity API</option>
                    <option value="openai_compatible" {selected("openai_compatible")}>OpenAI-compatible endpoint</option>
                  </optgroup>
                  <optgroup label="Developer tools">
                    <option value="preview" {selected("preview")}>Preview normalized input</option>
                  </optgroup>
                </select>
                <p class="mode-note">AI modes use keys from AI settings. Browse and offline work without keys.</p>
              </div>
              <div>
                <label for="view_mode_grid">Roadmap view</label>
                <div class="input-switch" id="view_toggle_group">
                  <label><input id="view_mode_grid" type="radio" name="view_mode" value="grid" {grid_checked}> Card grid</label>
                  <label><input type="radio" name="view_mode" value="list" {list_checked}> Compact list</label>
                </div>
              </div>
            </div>
            <div class="guided-panel active">
              <div class="wizard-launcher">
                {hidden_inputs_html}
                <div class="wizard-summary" id="wizard_summary">
                  {wizard_summary_html}
                </div>
                <div class="control-stack">
                  <button type="submit">Generate roadmap</button>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
      <div>
        {result_section}
      </div>
    </section>
    </section>
  </main>
  <div class="loading-shell" id="page_loading" aria-hidden="true">
    <div class="loading-bar-track"><div class="loading-bar-fill" id="page_loading_fill"></div></div>
    <div class="loading-copy" id="page_loading_copy">Building roadmap...</div>
  </div>
  <div class="modal-shell" id="settings_modal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="settings_title">
      <div class="modal-head">
        <div>
          <h2 id="settings_title">AI Settings</h2>
          <p>Save API keys locally in this browser so AI modes can run without manual env setup each time.</p>
        </div>
        <button type="button" class="modal-close" id="close_settings_button">Close</button>
      </div>
      <div class="settings-grid">
        <div class="settings-field"><label for="settings_openai_api_key">OpenAI API key</label><input id="settings_openai_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field"><label for="settings_gemini_api_key">Gemini API key</label><input id="settings_gemini_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field"><label for="settings_anthropic_api_key">Anthropic API key</label><input id="settings_anthropic_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field"><label for="settings_deepseek_api_key">DeepSeek API key</label><input id="settings_deepseek_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field"><label for="settings_perplexity_api_key">Perplexity API key</label><input id="settings_perplexity_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field"><label for="settings_openai_compatible_api_key">OpenAI-compatible key</label><input id="settings_openai_compatible_api_key" type="password" autocomplete="off"></div>
        <div class="settings-field wide"><label for="settings_openai_compatible_base_url">OpenAI-compatible base URL (optional)</label><input id="settings_openai_compatible_base_url" type="text" placeholder="http://127.0.0.1:11434/v1/chat/completions"></div>
        <div class="settings-field wide"><label for="settings_ollama_base_url">Ollama base URL (optional)</label><input id="settings_ollama_base_url" type="text" placeholder="http://127.0.0.1:11434"></div>
      </div>
      <p class="settings-help">Note: product subscriptions (for example ChatGPT Plus) are separate from API billing. Hosted API modes need a platform API key, while local modes like Ollama can run with no paid subscription.</p>
      <div class="settings-status" id="settings_status"></div>
      <div class="wizard-footer">
        <button type="button" class="wizard-secondary" id="clear_settings_button">Clear settings</button>
        <div class="wizard-footer-actions">
          <button type="button" id="save_settings_button">Save settings</button>
        </div>
      </div>
    </div>
  </div>
  <div class="modal-shell" id="wizard_modal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="wizard_title">
      <div class="modal-head">
        <div>
          <h2 id="wizard_title">New Path Wizard</h2>
          <p>Answer one question at a time. You can skip optional items and still generate a strong roadmap.</p>
        </div>
        <button type="button" class="modal-close" id="close_wizard_button">Close</button>
      </div>
      <div class="wizard-progress-panel">
        <div class="wizard-progress-bar" aria-hidden="true">
          <div class="wizard-progress-fill" id="wizard_progress_fill"></div>
        </div>
        <div class="wizard-progress-meta">
          <span class="wizard-progress-copy" id="wizard_progress_copy">Step 1 of 9</span>
          <div class="wizard-dots" id="wizard_progress_dots"></div>
        </div>
      </div>
      <div class="wizard-steps" id="wizard_steps">
        {wizard_steps_html}
      </div>
      <div class="wizard-status" id="wizard_status"></div>
      <div class="wizard-footer">
        <button type="button" class="wizard-secondary" id="wizard_back">Back</button>
        <div class="wizard-footer-actions">
          <button type="button" class="wizard-secondary" id="wizard_skip">Skip</button>
          <button type="button" id="wizard_next">Next</button>
          <button type="button" id="wizard_save" style="display:none;">Save path</button>
        </div>
      </div>
    </div>
  </div>
  <script>
    (() => {{
      const viewModeRadios = Array.from(document.querySelectorAll('input[name="view_mode"]'));
      const plannerForm = document.getElementById('planner_form');
      const weeksContainer = document.querySelector('.weeks-grid');
      const modal = document.getElementById('wizard_modal');
      const openButtons = Array.from(document.querySelectorAll('.open-wizard-trigger'));
      const closeButton = document.getElementById('close_wizard_button');
      const settingsModal = document.getElementById('settings_modal');
      const openSettingsButtons = Array.from(document.querySelectorAll('.open-settings-trigger'));
      const closeSettingsButton = document.getElementById('close_settings_button');
      const saveSettingsButton = document.getElementById('save_settings_button');
      const clearSettingsButton = document.getElementById('clear_settings_button');
      const settingsStatus = document.getElementById('settings_status');
      const steps = Array.from(document.querySelectorAll('.wizard-step'));
      const backButton = document.getElementById('wizard_back');
      const nextButton = document.getElementById('wizard_next');
      const skipButton = document.getElementById('wizard_skip');
      const saveButton = document.getElementById('wizard_save');
      const status = document.getElementById('wizard_status');
      const summary = document.getElementById('wizard_summary');
      const progressFill = document.getElementById('wizard_progress_fill');
      const progressCopy = document.getElementById('wizard_progress_copy');
      const progressDots = document.getElementById('wizard_progress_dots');
      const resourceProgressCard = document.getElementById('resource_progress_card');
      const resourceProgressFill = document.getElementById('resource_progress_fill');
      const resourceProgressPercent = document.getElementById('resource_progress_percent');
      const resourceProgressCopy = document.getElementById('resource_progress_copy');
      const resourceNextStep = document.getElementById('resource_next_step');
      const resourceChecks = Array.from(document.querySelectorAll('.resource-check'));
      const loadingShell = document.getElementById('page_loading');
      const loadingFill = document.getElementById('page_loading_fill');
      const loadingCopy = document.getElementById('page_loading_copy');
      const settingsStorageKey = 'learning_architect_ai_settings_v1';
      const settingsFields = [
        'openai_api_key',
        'gemini_api_key',
        'anthropic_api_key',
        'deepseek_api_key',
        'perplexity_api_key',
        'openai_compatible_api_key',
        'openai_compatible_base_url',
        'ollama_base_url'
      ];

      let currentStep = 0;
      const totalSteps = steps.length;
      let loadingTimer = null;
      const progressStorageKey = resourceProgressCard
        ? `learning_architect_progress::${{resourceProgressCard.dataset.progressKey || 'default'}}`
        : '';

      if (progressDots) {{
        progressDots.innerHTML = steps.map((_, index) => `<span class="wizard-dot" data-dot="${{index}}"></span>`).join('');
      }}

      function updateProgress() {{
        const progress = totalSteps ? ((currentStep + 1) / totalSteps) * 100 : 0;
        if (progressFill) progressFill.style.width = `${{progress}}%`;
        if (progressCopy) progressCopy.textContent = `Step ${{currentStep + 1}} of ${{totalSteps}}`;
        document.querySelectorAll('.wizard-dot').forEach((dot, index) => {{
          dot.classList.toggle('active', index === currentStep);
          dot.classList.toggle('complete', index < currentStep);
        }});
      }}

      function applyViewMode() {{
        if (!weeksContainer) return;
        const selected = document.querySelector('input[name="view_mode"]:checked')?.value || 'grid';
        weeksContainer.classList.toggle('list-view', selected === 'list');
      }}

      function startLoadingBar() {{
        if (!loadingShell || !loadingFill) return;
        let progress = 10;
        loadingShell.classList.add('active');
        loadingShell.setAttribute('aria-hidden', 'false');
        loadingFill.style.width = `${{progress}}%`;
        if (loadingCopy) loadingCopy.textContent = 'Building roadmap...';
        if (loadingTimer) window.clearInterval(loadingTimer);
        loadingTimer = window.setInterval(() => {{
          progress = Math.min(progress + (progress < 60 ? 8 : 3), 92);
          loadingFill.style.width = `${{progress}}%`;
        }}, 120);
      }}

      function readSettings() {{
        try {{
          return JSON.parse(window.localStorage.getItem(settingsStorageKey) || '{{}}');
        }} catch (error) {{
          return {{}};
        }}
      }}

      function writeSettings(value) {{
        try {{
          window.localStorage.setItem(settingsStorageKey, JSON.stringify(value));
        }} catch (error) {{
          return;
        }}
      }}

      function syncSettingsToHiddenInputs(settings) {{
        settingsFields.forEach((field) => {{
          const hidden = document.getElementById(`hidden_${{field}}`);
          if (hidden) hidden.value = settings[field] || '';
        }});
      }}

      function loadSettingsIntoModal() {{
        const settings = readSettings();
        settingsFields.forEach((field) => {{
          const input = document.getElementById(`settings_${{field}}`);
          if (input) input.value = settings[field] || '';
        }});
        syncSettingsToHiddenInputs(settings);
      }}

      function gatherSettingsFromModal() {{
        const settings = {{}};
        settingsFields.forEach((field) => {{
          const input = document.getElementById(`settings_${{field}}`);
          settings[field] = input ? input.value.trim() : '';
        }});
        return settings;
      }}

      function openSettingsModal() {{
        if (!settingsModal) return;
        loadSettingsIntoModal();
        settingsModal.classList.add('active');
        settingsModal.setAttribute('aria-hidden', 'false');
        if (settingsStatus) settingsStatus.textContent = '';
      }}

      function closeSettingsModal() {{
        if (!settingsModal) return;
        settingsModal.classList.remove('active');
        settingsModal.setAttribute('aria-hidden', 'true');
      }}

      function updateSummary() {{
        if (!summary) return;
        const fieldLabels = {{
          topic: 'Topic',
          experience_level: 'Experience level',
          schedule_length: 'Schedule length',
          time_available_per_week: 'Time per week',
          target_job_title: 'Target job title',
          job_industry_focus: 'Job or industry focus',
          domain_specialization: 'Domain specialization',
          secondary_goal: 'Secondary goal',
          custom_modifications: 'Custom modifications'
        }};
        const ordered = ['topic','experience_level','schedule_length','time_available_per_week','target_job_title','job_industry_focus','domain_specialization','secondary_goal','custom_modifications'];
        const chips = ordered
          .map((field) => {{
            const hidden = document.getElementById(`hidden_${{field}}`);
            const value = hidden ? hidden.value.trim() : '';
            if (!value) return '';
            return `<div class="summary-chip"><span class="summary-chip-label">${{fieldLabels[field]}}</span><strong>${{value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</strong></div>`;
          }})
          .filter(Boolean);
        summary.innerHTML = chips.length
          ? chips.join('')
          : "<p class='wizard-empty'>No guided answers yet. Start a new path and answer only what you know.</p>";
        openButtons.forEach((button) => {{
          button.textContent = chips.length ? 'Edit path answers' : 'Start new path';
        }});
      }}

      function syncHiddenInputs() {{
        document.querySelectorAll('.wizard-control').forEach((control) => {{
          const field = control.dataset.field;
          const hidden = document.getElementById(`hidden_${{field}}`);
          if (hidden) hidden.value = control.value;
        }});
        updateSummary();
      }}

      function showStep(index) {{
        currentStep = Math.max(0, Math.min(index, steps.length - 1));
        steps.forEach((step, idx) => step.classList.toggle('active', idx === currentStep));
        updateProgress();
        if (backButton) backButton.style.visibility = currentStep === 0 ? 'hidden' : 'visible';
        const isLast = currentStep === steps.length - 1;
        if (nextButton) nextButton.style.display = isLast ? 'none' : 'inline-flex';
        if (skipButton) skipButton.style.display = isLast ? 'none' : 'inline-flex';
        if (saveButton) saveButton.style.display = isLast ? 'inline-flex' : 'none';
        if (status) status.textContent = '';
        const activeControl = steps[currentStep]?.querySelector('.wizard-control');
        if (activeControl) activeControl.focus();
      }}

      function readStoredChecklist() {{
        if (!progressStorageKey) return {{}};
        try {{
          return JSON.parse(window.localStorage.getItem(progressStorageKey) || '{{}}');
        }} catch (error) {{
          return {{}};
        }}
      }}

      function writeStoredChecklist(value) {{
        if (!progressStorageKey) return;
        try {{
          window.localStorage.setItem(progressStorageKey, JSON.stringify(value));
        }} catch (error) {{
          return;
        }}
      }}

      function updateResourceProgress() {{
        if (!resourceProgressCard || !resourceChecks.length) return;
        const stored = readStoredChecklist();
        const total = resourceChecks.length;
        let completed = 0;
        let nextItem = null;
        const weekCounts = {{}};

        resourceChecks.forEach((checkbox, index) => {{
          const itemKey = checkbox.dataset.progressItem;
          const checked = Boolean(stored[itemKey]);
          const item = checkbox.closest('.resource-item');
          const week = checkbox.dataset.week;
          checkbox.checked = checked;
          if (item) {{
            item.classList.toggle('is-complete', checked);
            item.classList.remove('is-next');
          }}
          if (!weekCounts[week]) {{
            weekCounts[week] = {{ total: 0, done: 0 }};
          }}
          weekCounts[week].total += 1;
          if (checked) {{
            completed += 1;
            weekCounts[week].done += 1;
          }} else if (!nextItem) {{
            nextItem = {{ checkbox, index }};
          }}
        }});

        if (nextItem) {{
          const nextElement = nextItem.checkbox.closest('.resource-item');
          if (nextElement) nextElement.classList.add('is-next');
        }}

        const percent = total ? Math.round((completed / total) * 100) : 0;
        if (resourceProgressFill) resourceProgressFill.style.width = `${{percent}}%`;
        if (resourceProgressPercent) resourceProgressPercent.textContent = `${{percent}}%`;
        if (resourceProgressCopy) {{
          resourceProgressCopy.textContent = `${{completed}} of ${{total}} resources completed.`;
        }}
        if (resourceNextStep) {{
          if (nextItem) {{
            const week = nextItem.checkbox.dataset.week;
            const resourceNumber = nextItem.checkbox.dataset.resourceNumber || '1';
            resourceNextStep.textContent = `Next up: Week ${{week}}, Resource ${{resourceNumber}}`;
          }} else {{
            resourceNextStep.textContent = 'All listed resources completed';
          }}
        }}

        document.querySelectorAll('[data-week-progress]').forEach((pill) => {{
          const week = pill.dataset.weekProgress;
          const counts = weekCounts[week] || {{ total: 0, done: 0 }};
          const weekPercent = counts.total ? Math.round((counts.done / counts.total) * 100) : 0;
          pill.textContent = `${{weekPercent}}% complete`;
        }});

        const pathSteps = Array.from(document.querySelectorAll('[data-path-week]'));
        let anyActive = false;
        pathSteps.forEach((step) => {{
          const week = step.dataset.pathWeek;
          const counts = weekCounts[week] || {{ total: 0, done: 0 }};
          const weekPercent = counts.total ? Math.round((counts.done / counts.total) * 100) : 0;
          step.classList.remove('active', 'complete');
          if (weekPercent >= 100) {{
            step.classList.add('complete');
          }} else if (weekPercent > 0) {{
            step.classList.add('active');
            anyActive = true;
          }}
        }});
        if (!anyActive) {{
          const firstPending = pathSteps.find((step) => !step.classList.contains('complete'));
          if (firstPending) firstPending.classList.add('active');
        }}
      }}

      function handleResourceCheck(event) {{
        const checkbox = event.currentTarget;
        const itemKey = checkbox.dataset.progressItem;
        const stored = readStoredChecklist();
        stored[itemKey] = checkbox.checked;
        writeStoredChecklist(stored);
        updateResourceProgress();
      }}

      function openWizard() {{
        if (!modal) return;
        modal.classList.add('active');
        modal.setAttribute('aria-hidden', 'false');
        showStep(currentStep);
      }}

      function closeWizard() {{
        if (!modal) return;
        modal.classList.remove('active');
        modal.setAttribute('aria-hidden', 'true');
      }}

      function requireCurrentIfNeeded() {{
        const step = steps[currentStep];
        if (!step) return true;
        const field = step.dataset.field;
        const control = step.querySelector('.wizard-control');
        if (!control) return true;
        if (field === 'topic' && !control.value.trim()) {{
          if (status) status.textContent = 'Topic is the only required answer to build a path.';
          control.focus();
          return false;
        }}
        return true;
      }}

      viewModeRadios.forEach((radio) => radio.addEventListener('change', applyViewMode));
      openButtons.forEach((button) => button.addEventListener('click', openWizard));
      openSettingsButtons.forEach((button) => button.addEventListener('click', openSettingsModal));
      if (closeButton) closeButton.addEventListener('click', closeWizard);
      if (closeSettingsButton) closeSettingsButton.addEventListener('click', closeSettingsModal);
      if (modal) {{
        modal.addEventListener('click', (event) => {{
          if (event.target === modal) closeWizard();
        }});
      }}
      if (settingsModal) {{
        settingsModal.addEventListener('click', (event) => {{
          if (event.target === settingsModal) closeSettingsModal();
        }});
      }}
      if (saveSettingsButton) {{
        saveSettingsButton.addEventListener('click', () => {{
          const settings = gatherSettingsFromModal();
          writeSettings(settings);
          syncSettingsToHiddenInputs(settings);
          if (settingsStatus) settingsStatus.textContent = 'Settings saved locally for this browser.';
          window.setTimeout(() => {{
            closeSettingsModal();
          }}, 350);
        }});
      }}
      if (clearSettingsButton) {{
        clearSettingsButton.addEventListener('click', () => {{
          const empty = settingsFields.reduce((acc, field) => (acc[field] = '', acc), {{}});
          writeSettings(empty);
          loadSettingsIntoModal();
          if (settingsStatus) settingsStatus.textContent = 'Saved settings cleared.';
        }});
      }}
      if (backButton) backButton.addEventListener('click', () => showStep(currentStep - 1));
      if (skipButton) skipButton.addEventListener('click', () => showStep(currentStep + 1));
      if (nextButton) nextButton.addEventListener('click', () => {{
        if (!requireCurrentIfNeeded()) return;
        syncHiddenInputs();
        showStep(currentStep + 1);
      }});
      if (saveButton) saveButton.addEventListener('click', () => {{
        if (!requireCurrentIfNeeded()) return;
        syncHiddenInputs();
        closeWizard();
        if (plannerForm) {{
          if (typeof plannerForm.requestSubmit === 'function') {{
            plannerForm.requestSubmit();
          }} else {{
            plannerForm.submit();
          }}
        }}
      }});
      if (plannerForm) {{
        plannerForm.addEventListener('submit', () => {{
          syncSettingsToHiddenInputs(readSettings());
          startLoadingBar();
        }});
      }}
      document.querySelectorAll('.wizard-control').forEach((control) => {{
        control.addEventListener('input', syncHiddenInputs);
        control.addEventListener('keydown', (event) => {{
          if (event.key === 'Enter' && control.tagName !== 'TEXTAREA' && currentStep < steps.length - 1) {{
            event.preventDefault();
            if (!requireCurrentIfNeeded()) return;
            syncHiddenInputs();
            showStep(currentStep + 1);
          }}
        }});
      }});
      resourceChecks.forEach((checkbox) => checkbox.addEventListener('change', handleResourceCheck));

      applyViewMode();
      loadSettingsIntoModal();
      syncHiddenInputs();
      showStep(0);
      updateResourceProgress();
    }})();
  </script>
</body>
</html>"""


class LearningArchitectHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send_html(render_page(_default_form_state()))

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(content_length).decode("utf-8"))
        form_state = _merge_form_state({key: values[0] for key, values in form.items() if values})
        mode = form_state.get("mode", "browse")

        try:
            user_input = _build_user_input_from_state(form_state)
            api_key_field = _api_key_field_for_mode(mode)
            api_key = form_state.get(api_key_field, "").strip() if api_key_field else ""
            env_overrides = _env_overrides_from_form_state(form_state)
            if mode == "preview":
                body = render_page(
                    form_state=form_state,
                    preview_json=build_ai_messages(user_input),
                )
            else:
                with _temporary_env(env_overrides):
                    body = render_page(
                        form_state=form_state,
                        result=generate_roadmap(
                            user_input=user_input,
                            provider=mode,
                            api_key=api_key or None,
                        ),
                    )
        except Exception as exc:
            body = render_page(
                form_state=form_state,
                error=str(exc),
            )

        self._send_html(body)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    preferred_port = int(os.getenv("PORT", "8000"))
    for port in range(preferred_port, preferred_port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), LearningArchitectHandler)
            print(f"Serving AI Learning Architect on http://127.0.0.1:{port}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down AI Learning Architect.")
            finally:
                server.server_close()
            return
        except OSError as exc:
            if exc.errno != 10048:
                raise
            continue
    raise OSError("Could not bind to a local port between 8000 and 8009.")


if __name__ == "__main__":
    main()
