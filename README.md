# DocuMind AI

DocuMind AI is an evidence-first document assistant. Upload text-based PDFs, ask natural-language questions, and receive grounded answers with document and page citations.

## What is included

- PDF type, size, signature, readability, and empty-text validation
- Page-aware text extraction and overlapping chunks
- OpenAI `text-embedding-3-small` embeddings stored in persistent ChromaDB
- Semantic search across multiple documents
- Grounded answer generation with page-level citations and unknown-answer handling
- Document listing and deletion
- Streamlit chat interface with conversation history
- FastAPI endpoints, automated API tests, and Docker deployment
- Prompt-injection defense: document text is explicitly treated as untrusted evidence

## Architecture

```text
Browser → Streamlit → FastAPI → PyMuPDF → ChromaDB
                            ↘ OpenAI embeddings + chat model
```

The backend owns document processing and retrieval. Streamlit is a thin API client, which keeps the interface replaceable and makes API testing straightforward.

## Run locally

Prerequisites: Python 3.12 and an OpenAI API key with available credits.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Add your key to `.env`, then run these commands in separate terminals:

```bash
make api
make ui
```

Open `http://localhost:8501`. API documentation is available at `http://localhost:8000/docs`.

## Run with Docker

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env
docker compose up --build
```

The application stores uploads and the Chroma index in a named Docker volume.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Health and configuration status |
| POST | `/documents/upload` | Validate and index one PDF |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document and its chunks |
| POST | `/ask` | Retrieve evidence and generate a cited answer |

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the vacation policy?"}'
```

## Tests

```bash
make test
```

The API tests replace the external AI/vector service with a deterministic fake, so they run without credits or network access. Before a public launch, add authentication, per-user document isolation, malware scanning, rate limits, and at least 30 domain-specific evaluation questions.

## Suggested portfolio evaluation

Create a CSV with question, expected document/page, expected answer facts, retrieved pages, answer, citations, and latency. Report retrieval hit rate, answer correctness, citation correctness, refusal correctness, and p50/p95 latency. Keep the evaluation PDFs and expected outputs versioned, but never commit private documents or API keys.

