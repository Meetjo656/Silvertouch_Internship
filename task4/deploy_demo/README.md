# Text-to-SQL + RAG demo

A lightweight, deployable demo built alongside the original Task 4 research pipeline. It intentionally uses a **synthetic retail database**; it does not load or expose the internship's exported enterprise data.

## Features

- Lightweight retrieval over table descriptions and business rules.
- OpenAI-compatible chat completion endpoint for SQL generation.
- SQL parsing and read-only SELECT enforcement with SQLGlot.
- SQLite execution in read-only mode, capped at 100 rows.
- Browser UI showing retrieved context, SQL, validation and results.

## Run locally

```bash
cd task4/deploy_demo
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set a model API key. OpenAI example:

```bash
# PowerShell
$env:OPENAI_API_KEY="your-key"
$env:MODEL_NAME="gpt-4o-mini"
# macOS/Linux
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
```

Groq is also supported:

```bash
export GROQ_API_KEY="your-key"
export MODEL_NAME="llama-3.3-70b-versatile"
```

Then start the server:

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Deploy on Render

1. Create a new **Web Service** connected to this repository.
2. Set the Root Directory to `task4/deploy_demo`.
3. Render detects `render.yaml`; otherwise use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add `OPENAI_API_KEY` or `GROQ_API_KEY` as a secret environment variable.
5. Optionally set `MODEL_NAME` and `OPENAI_BASE_URL`.

The SQLite database is generated from synthetic seed data at startup. On free ephemeral hosting, demo changes are not persistent.

## Important limitations

- This is a public portfolio demo, not a production enterprise deployment.
- The sample dataset is synthetic and small.
- Retrieval is lexical in this first deployable version; it is deliberately lightweight and avoids downloading embedding models at startup.
- The model provider is external and requires the deployer to supply an API key.
- SQL execution is read-only and restricted to the four sample tables.
- Do not add private exports, credentials, customer records, or proprietary internship data to this demo.
