# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload **any file** — documents, images, audio, or video — and ask questions about it. Answers are generated using a local LLM with cited references.

**Course:** COMP-4400 Programming LLMs — University of Windsor

## Supported File Types

| Category | Formats | How it works |
|----------|---------|--------------|
| Documents | PDF, DOCX, PPTX, TXT, MD, CSV, HTML, JSON | Text extraction |
| Images | PNG, JPG, JPEG, WEBP | OCR via EasyOCR |
| Audio/Video | MP3, WAV, MP4, MOV, WEBM, M4A | Speech-to-text via Whisper |

## Features

- **Multi-format support** — 18 file types across documents, images, and media
- **Image OCR** — Extract text from screenshots, scanned docs, photos of whiteboards
- **Audio/Video transcription** — Transcribe lectures, meetings, podcasts, then Q&A over them
- **Semantic Chunking** — Overlapping chunks for accurate retrieval
- **Vector Search** — Embed chunks with `sentence-transformers` and store in ChromaDB
- **Cross-Encoder Reranking** — Two-stage retrieval for higher precision
- **Local LLM Generation** — Ollama (no API keys, no cost, fully private)
- **Source Citations** — Every answer shows which sections were used
- **Chat Interface** — Multi-turn conversation with full history

## Architecture

```
File Upload → Text Extraction / OCR / Transcription → Chunking → Embedding → ChromaDB
                                                                                  ↓
User Query → Query Embedding → Vector Search → Cross-Encoder Rerank
                                                                                  ↓
                                                      Top-K Chunks + Query → Ollama LLM → Answer with Citations
```

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | [Ollama](https://ollama.com) (local) | Free |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free |
| Reranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Free |
| Vector DB | ChromaDB (in-memory) | Free |
| OCR | EasyOCR | Free |
| Transcription | OpenAI Whisper (local) | Free |
| UI | Streamlit | Free |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Naitya07/rag-document-qa.git
cd rag-document-qa
```

### 2. Install Ollama

Download from [ollama.com](https://ollama.com) and install it. Then pull the model:

```bash
ollama pull llama3.2
```

### 3. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the app

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Upload any supported file using the sidebar
2. Click "Process Document" to extract text, chunk, and embed
3. Ask questions in the chat input
4. View answers with expandable source citations

### Examples

- Upload a **PDF textbook chapter** → "What are the key concepts in chapter 3?"
- Upload a **lecture recording (.mp4)** → "What did the professor say about transformers?"
- Upload a **screenshot of code (.png)** → "What does this function do?"
- Upload a **CSV dataset** → "Summarize the data in this file"

## Project Structure

```
rag-document-qa/
├── app.py               # Streamlit UI and chat logic
├── rag/
│   ├── __init__.py
│   ├── chunker.py       # Multi-format text extraction, OCR, transcription, chunking
│   ├── embedder.py      # Sentence-transformer embeddings
│   ├── retriever.py     # ChromaDB storage + cross-encoder reranking
│   └── generator.py     # Ollama LLM generation
├── requirements.txt
├── .gitignore
└── README.md
```

## Course Topics Covered

- **APIs** — Programmatic LLM interaction via Ollama API
- **Prompting** — System prompt design with role, context injection, and temperature control
- **RAG** — Full retrieval-augmented generation pipeline
- **Cross-Encoder Reranking** — Two-stage retrieval for improved precision

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request
