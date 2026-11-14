# Text-to-SQL + RAG demo

A lightweight, deployable demo built alongside the original Task 4 research pipeline. It intentionally uses a **synthetic retail database**; it does not load or expose the internship's exported enterprise data.

## Features

- Lightweight retrieval over table descriptions and business rules.
- Groq-hosted chat completion for SQL generation (OpenAI-compatible endpoint).
- SQL parsing and read-only SELECT enforcement with SQLGlot.
- SQLite execution in read-only mode, capped at 100 rows.
- Browser UI showing retrieved context, SQL, validation and results.

## Run locally

```bash
cd task4/deploy_demo
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a Groq API key in the [Groq Console](https://console.groq.com/keys). Keep it private and never commit it to GitHub.

Set the environment variables:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-key"
$env:MODEL_NAME="openai/gpt-oss-20b"

# macOS/Linux
export GROQ_API_KEY="your-key"
export MODEL_NAME="openai/gpt-oss-20b"
```

The default model is `openai/gpt-oss-20b`. You can change `MODEL_NAME` to another currently available Groq chat model from the [Groq model catalogue](https://console.groq.com/docs/models).

Then start the server:

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Deploy on Render

1. Create a new **Web Service** connected to this repository.
2. Set the Root Directory to `task4/deploy_demo`.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add `GROQ_API_KEY` as a secret environment variable in Render.
5. Set `MODEL_NAME` to `openai/gpt-oss-20b` (or another model currently enabled for your Groq account).

The SQLite database is generated from synthetic seed data at startup. On free ephemeral hosting, demo changes are not persistent.

## Important limitations

- This is a public portfolio demo, not a production enterprise deployment.
- The sample dataset is synthetic and small.
- Retrieval is lexical in this first deployable version; it is deliberately lightweight and avoids downloading embedding models at startup.
- Groq provides the model inference endpoint; the deployer supplies the API key.
- SQL execution is read-only and restricted to the four sample tables.
- Do not add private exports, credentials, customer records, or proprietary internship data to this demo.
