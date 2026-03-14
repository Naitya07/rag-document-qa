"""LLM generation via Ollama (local) API."""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def generate_answer(query: str, context_chunks: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Send query + retrieved context to Ollama and return the answer."""
    context = "\n\n".join(
        f"[{c.get('source', 'doc')} — Section {c['page']}] {c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful document assistant. Answer the question based on the provided context.
Use the context to give the best possible answer. For broad questions like "what is this about" or "summarize",
provide an overview based on all the context you can see.
Always cite the source file name and section number you used.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running (`ollama serve`)."
    except requests.HTTPError as e:
        return f"Error from Ollama: {e}"
    except Exception as e:
        return f"Error: {e}"
