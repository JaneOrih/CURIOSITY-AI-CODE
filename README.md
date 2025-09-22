# Curiosity AI

Curiosity AI is a demonstration project that orchestrates large language models (LLMs) to explore a topic in an open‑ended yet controlled manner.  
It generates expert‑level questions, measures their novelty relative to a small corpus, hunts for contradictions using natural language inference (NLI) models, and exposes its reasoning process through a simple API and user interface.

## Features

* **Question generation** – Given a seed topic, the engine proposes a batch of concise questions using an underlying LLM.  
* **Novelty scoring** – Each question is scored by comparing its embedding to those of existing knowledge snippets stored in a FAISS index; higher scores indicate more novel inquiries.
* **Bounded exploration** – The engine continues to generate fresh questions until novelty drops below a threshold or a time limit is reached.
* **Contradiction detection** – Pairs of questions are evaluated with an NLI model to identify contradictory statements.
* **REST API** – A FastAPI endpoint (`/ask`) accepts a seed topic and returns a curiosity trail and dissonance log.
* **Streamlit UI** – A simple web front‑end visualises the questions and contradictions discovered during exploration.

## Repository Layout

The repository is laid out to separate concerns between API, engine logic, and user interface.

```
Curiosity-AI/
├── app/                         # FastAPI application
│   └── main.py                 # API entry point
├── agents/                     # High level agent orchestrator
│   └── orchestrator.py         # Coordinates the curiosity engine
├── engine/                     # Core engine components
│   ├── curiosity_engine.py     # Question generation, novelty scoring, bounded exploration
│   ├── models.py               # LLM router abstraction
│   ├── nli_contradiction.py    # Contradiction detection via NLI models
│   └── retrieval.py            # FAISS retrieval and search helpers
├── ui/                         # Streamlit user interface
│   └── streamlit_app.py        # Front‑end entry point
├── scripts/                    # Helper scripts
│   └── build_vectorstore.py    # Build FAISS index from text corpus
├── config/                     # Configuration files
│   └── config.yaml             # Model and engine configuration
├── data/                       # Example text corpus
│   └── sample.txt              # Seed data used to build the index
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview (this file)
```

## Getting Started

### Prerequisites

* Python 3.10 or newer
* [Ollama](https://ollama.com/) (optional, for running local models)
* A CUDA‑enabled GPU (optional, only required for large models)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Curiosity-AI
   ```

2. **Install dependencies:**

   It is recommended to create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download language models (optional):**

   If you want to use local models via Ollama, install Ollama and pull the desired models:

   ```bash
   ollama pull llama3
   ollama pull qwen2.5:7b-instruct
   ```

4. **Build the FAISS index:**

   Before running the API you need to build a vector index from your data.  
   The following command reads all `.txt` files in the `data/` directory, embeds them using a sentence transformer and writes a FAISS index to `config/index.faiss`.

   ```bash
   python scripts/build_vectorstore.py --data-dir data --index-path config/index.faiss
   ```

### Running the API

To start the FastAPI server with hot reloading:

```bash
uvicorn app.main:app --reload --port 8000
```

Once running, you can send a POST request to `http://localhost:8000/ask` with JSON payload `{"topic": "Your seed topic"}`.  
The response will contain a trail of generated questions and a dissonance log of detected contradictions.

### Running the Streamlit UI

To start the Streamlit app on port 8501:

```bash
streamlit run ui/streamlit_app.py
```

Enter a seed topic and press “Explore” to view the generated curiosity trail and contradictions in a browser.

## Configuration

The behaviour of Curiosity AI is controlled via the YAML file at `config/config.yaml`.  
Key options include:

* `models.primary` – The default LLM to use.  
  Format is `<provider>:<model_name>`, e.g. `openai:gpt-4o-mini` or `ollama:llama3`.
* `models.nli` – The name of the NLI model used for contradiction detection (defaults to `roberta-large-mnli`).
* `retrieval.embedder` – Sentence transformer model used to embed text snippets.
* `engine.max_rounds` – Maximum number of exploration cycles per session.
* `engine.novelty_threshold` – Minimum novelty score to accept a question.

Edit this file to switch models or tweak the exploration parameters.  
If you are using remote APIs (OpenAI, Anthropic), you should set the corresponding API keys as environment variables before running the server:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

## Notes

* This project is a proof of concept.  It is not optimised for production use and should be treated as experimental software.
* The sample data in the `data` directory is deliberately small; to get meaningful results you should replace it with a more substantial corpus relevant to your domain.
* The system is designed to be modular.  You can swap out the underlying language models, retrieval strategy or scoring functions without changing the overall architecture.

## License

This project is provided under the MIT license.  See the `LICENSE` file for details.