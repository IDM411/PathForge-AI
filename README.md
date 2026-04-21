# PathForge AI

Open-source learning roadmap generator focused on practical, job-aligned outcomes with free resources first.
It works without paid AI by default, while still letting users plug in their own provider when they want it.

## Why This Exists

Most roadmap tools are either too generic or too locked behind paid AI.
Learning Path AI is designed to stay accessible, practical, and current by combining local planning logic with live public-web resource discovery.

## Core Features

- Free-first resource strategy with quality and timeline checks.
- Weekly, modular roadmaps aligned to job goals and specializations.
- Live non-AI web discovery from trusted learning domains.
- Direct launchable resource links with durable direct-site fallback.
- Guided browser wizard plus progress checklist and completion bar.
- Optional bring-your-own AI providers: Ollama, OpenAI, Gemini, Anthropic, DeepSeek, Perplexity, and OpenAI-compatible endpoints.

## Quick Start

Requirements:

- Python 3.10+

Run the browser app:

```powershell
python app.py
```

Open `http://127.0.0.1:8000`, click `Start new path`, answer the prompts, and generate your roadmap.

## CLI Usage

Default mode (`browse`) uses live non-AI discovery:

```powershell
python learning_architect.py demo_request.json
```

Offline-only mode:

```powershell
python learning_architect.py demo_request.json --provider offline
```

You can also pass plain text via stdin:

```powershell
"Machine Learning for MLOps" | python learning_architect.py
```

Preview normalized AI messages:

```powershell
python learning_architect.py demo_request.json --preview-ai-messages
```

Preview OpenAI request payload:

```powershell
python learning_architect.py demo_request.json --preview-openai-payload
```

## Optional AI Providers

Example commands:

```powershell
python learning_architect.py demo_request.json --provider openai
python learning_architect.py demo_request.json --provider ollama --model llama3.1:8b
python learning_architect.py demo_request.json --provider gemini
python learning_architect.py demo_request.json --provider anthropic
python learning_architect.py demo_request.json --provider deepseek
python learning_architect.py demo_request.json --provider perplexity
python learning_architect.py demo_request.json --provider openai_compatible --model your-model
```

Environment variables:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`
- `PERPLEXITY_API_KEY`
- `OLLAMA_BASE_URL` (optional, defaults to `http://127.0.0.1:11434`)
- `OPENAI_COMPATIBLE_BASE_URL`
- `OPENAI_COMPATIBLE_API_KEY`
- `OPENAI_COMPATIBLE_MODEL`

## Input Shape

Supports either a plain topic string or a JSON object.
Common fields include:

- `topic`
- `experience_level`
- `schedule_length`
- `time_available_per_week`
- `target_job_title`
- `job_industry_focus`
- `domain_specialization`
- `secondary_goal`
- `custom_modifications`

Common aliases such as `subject`, `level`, and `hours per week` are normalized automatically.

## Resource Quality Policy

- Prioritize high-trust free sources (official docs, freeCodeCamp, MIT OCW, Khan Academy, fast.ai, GitHub, selected YouTube).
- Reject weak or paid-signaling hits when stronger free options are available.
- Enforce weekly time caps by time-boxing or replacing overly long resources with focused alternatives.

## Known Limitations

Some topic combinations and fallback pairings are still being refined, so a few edge-case roadmaps may not align perfectly yet.
This is an active area of cleanup and will be improved in later updates.

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Project Structure

- `app.py`: Browser UI server and interactive workflow.
- `learning_architect.py`: Roadmap generation, normalization, review, and provider dispatch.
- `resource_discovery.py`: Live non-AI discovery and trusted-domain ranking.
- `tests/`: Unit tests for UI rendering, roadmap logic, and discovery behavior.

## Contributing

Issues and pull requests are welcome.
If you contribute, prioritize free-accessible resources, realistic weekly scope, and clear job relevance.
