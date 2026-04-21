# AI Learning Architect

This project now has a no-AI live browsing path as its default behavior, plus optional bring-your-own AI integrations.

The roadmap is generated locally from curated topic profiles and scheduling logic. Resource freshness comes from direct public web discovery, not an AI provider or paid search API, so anyone can use it without an API key. If you want AI assistance, you can plug in your own local or hosted provider instead of being locked to one vendor.

## Run it

Browser app:

```powershell
python app.py
```

The browser UI now starts with a blank guided flow instead of a prefilled demo request. You can either:

- click `Start new path` and answer questions one at a time in the popup wizard
- switch to raw JSON or plain-text input if you prefer
- switch between `Card grid` and `Compact list` roadmap views for easier scanning on longer plans

CLI with live non-AI browsing:

```powershell
python learning_architect.py demo_request.json
```

CLI with offline-only fallback:

```powershell
python learning_architect.py demo_request.json --provider offline
```

Preview the OpenAI request payload:

```powershell
python learning_architect.py demo_request.json --provider offline --preview-openai-payload
```

Optional OpenAI path:

```powershell
$env:OPENAI_API_KEY="your_key_here"
python learning_architect.py demo_request.json --provider openai
```

Optional local open-source path with Ollama:

```powershell
ollama serve
python learning_architect.py demo_request.json --provider ollama --model llama3.1:8b
```

Other optional AI providers:

```powershell
python learning_architect.py demo_request.json --provider gemini
python learning_architect.py demo_request.json --provider anthropic
python learning_architect.py demo_request.json --provider deepseek
python learning_architect.py demo_request.json --provider perplexity
python learning_architect.py demo_request.json --provider openai_compatible --model your-model
```

## What changed

- `browse` is still the default provider
- live resource discovery uses public DuckDuckGo HTML search and trusted-domain scoring, with no API key required
- the app renders weekly cards with clickable resources and live or fallback badges
- offline generation still works if live browsing cannot fetch strong results
- AI is now optional and pluggable: `ollama`, `openai`, `gemini`, `anthropic`, `deepseek`, `perplexity`, and `openai_compatible`

## Resource discovery

The live discovery layer looks for fresh resources from the public web and prefers trusted free-learning domains such as:

- official documentation sites
- GitHub
- YouTube
- freeCodeCamp
- MIT OpenCourseWare
- Khan Academy
- Coursera free pages
- fast.ai

If a strong live result cannot be fetched, the app falls back to a durable search link instead of failing.

## AI provider environment variables

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPSEEK_API_KEY`
- `PERPLEXITY_API_KEY`
- `OLLAMA_BASE_URL` for local Ollama if not using `http://127.0.0.1:11434`
- `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, and `OPENAI_COMPATIBLE_MODEL` for any compatible local or hosted endpoint

The AI integrations use your own access. If you do not have keys, the browse and offline modes still work fully.

## Input shape

You can pass either:

- A plain topic string like `Machine Learning`
- A JSON object containing any mix of:
  - `topic`
  - `experience_level`
  - `schedule_length`
  - `job_industry_focus`
  - `custom_modifications`
  - `secondary_goal`
  - `domain_specialization`
  - `time_available_per_week`
  - `target_job_title`

Common aliases such as `subject`, `level`, `hours per week`, and `industry focus` are normalized automatically.
