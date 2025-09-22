"""
FastAPI application entry point for the Curiosity AI service.

This module initialises the retrieval, curiosity engine and orchestrator
using the configuration defined in `config/config.yaml`.  It exposes a single
endpoint `/ask` which accepts a seed topic and returns a curiosity trail of
questions along with any detected contradictions.

To run the server locally:

```
uvicorn app.main:app --reload --port 8000
```

If you are using remote LLM providers (OpenAI, Anthropic) ensure that the
corresponding API keys are set in your environment variables.
"""
from __future__ import annotations

import asyncio
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.retrieval import Retriever
from engine.curiosity_engine import CuriosityEngine
from agents.orchestrator import Orchestrator


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import


app = FastAPI(title="Curiosity AI API", version="0.1")

# ADD THIS SECTION
# Configure Cross-Origin Resource Sharing (CORS) so browsers can call the API
origins = ["*"]  # For development allow all origins; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load configuration at startup.  The YAML file contains model specifications,
# retrieval settings and engine parameters.  Any missing keys will raise
# errors during initialisation.
def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = _load_config("config/config.yaml")

# Instantiate the retrieval system.  The Retriever loads a FAISS index and
# accompanying metadata from disk; if the index does not exist you must
# generate it using scripts/build_vectorstore.py before starting the API.
retriever = Retriever(
    embedder_name=cfg["retrieval"]["embedder"],
    index_path=cfg["retrieval"]["index_path"],
    graph_path=cfg["retrieval"].get("graph_path", "")
)

# Instantiate the curiosity engine with model settings, retrieval and NLI.
engine = CuriosityEngine(
    model_spec=cfg["models"]["primary"],
    retriever=retriever,
    nli_name=cfg["models"]["nli"],
    novelty_threshold=cfg["engine"]["novelty_threshold"],
    max_round_seconds=cfg["engine"]["max_round_seconds"],
    max_rounds=cfg["engine"]["max_rounds"]
)

# High level orchestrator wraps the engine and adds dissonance detection.
orchestrator = Orchestrator(engine)


class AskRequest(BaseModel):
    """Schema for incoming requests to the `/ask` endpoint."""

    topic: str


class AskResponse(BaseModel):
    """Schema for responses returned by the `/ask` endpoint."""

    topic: str
    session: dict
    dissonance: list


app = FastAPI(title="Curiosity AI API", version="0.1")


@app.get("/health")
def health() -> dict:
    """Simple health check to verify that the service is running."""

    return {"status": "ok"}

@app.get("/")
def health() -> dict:
    """Home Page."""

    return {"status": "Welcome"}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """
    Generate a curiosity trail of questions for a given topic.

    The request body must contain a `topic` field with a string describing
    what the user is interested in.  The engine will perform a bounded
    exploration starting from this seed topic and return a list of questions
    ranked by novelty, along with any contradictory pairs of statements.

    If the engine fails to produce any questions an HTTP error is raised.
    """
    if not req.topic or not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must be a non-empty string.")
    try:
        # The orchestrator returns a dictionary with the original topic,
        # session information (including the question trail) and dissonance entries.
        result = await orchestrator.run_cycle(req.topic)
    except Exception as e:  # pragma: no cover - catch unforeseen errors
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")
    return AskResponse(**result)