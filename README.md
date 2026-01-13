# Enterprise Knowledge Assistant

An intelligent RAG (Retrieval-Augmented Generation) system that processes official FastAPI documentation and provides:
- **Document Summarization** - Section and executive summaries
- **FAQ Generation** - Automatically generated Q&A pairs
- **Question Answering** - RAG-based answers with source attribution
- **Knowledge Grounding** - Says "The knowledge base does not contain this information." when needed

## 📚 Documentation Used

**FastAPI Official Documentation** - https://fastapi.tiangolo.com/tutorial/

The system ingests 10 tutorial pages covering:
- First Steps
- Path Parameters
- Query Parameters
- Request Body
- Query/Path Validation
- Multiple Body Parameters
- Response Models
- Extra Models
- Response Status Codes

## 🏗️ Architecture

```
enterprise_knowledge_assistant/
│
├── docs_ingestion/           # Document ingestion pipeline
│   ├── scrape_docs.py        # Web scraper (requests + BeautifulSoup)
│   ├── clean_text.py         # Text preprocessing
│   └── chunk_docs.py         # Document chunking
│
├── rag_pipeline/             # RAG components
│   ├── embeddings.py         # SentenceTransformers embeddings
│   ├── vector_store.py       # FAISS vector database
│   ├── retriever.py          # Semantic search retrieval
│   └── qa.py                 # LLM-based Q&A engine
│
├── summarization/            # Content generation
│   ├── section_summary.py    # Per-document summaries
│   ├── executive_summary.py  # High-level overview
│   └── faq_generator.py      # FAQ generation
│
├── docs/                     # Document storage
│   ├── raw/                  # Scraped raw documents
│   └── processed/            # Cleaned documents
│
├── data/                     # Generated data
│   ├── chunks.json           # Document chunks
│   ├── embeddings.npy        # Vector embeddings
│   ├── faiss_index.bin       # FAISS index
│   └── *.json                # Summaries, FAQs
│
├── main.py                   # CLI application
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Features

### 1. Document Ingestion
- Scrapes FastAPI tutorial pages using `requests` + `BeautifulSoup`
- Cleans text (removes navigation, short lines, noise)
- Chunks documents into 500-word segments with overlap

### 2. Embeddings & Vector Store
- Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for embeddings
- Stores vectors in `FAISS` for fast similarity search
- Supports cosine similarity with configurable threshold

### 3. Summarization
- **Section Summaries**: 100-word summaries per document
- **Executive Summary**: 300-word high-level overview
- All summaries strictly grounded in source documentation

### 4. FAQ Generation
- Generates 10 developer-focused Q&A pairs
- Each FAQ includes source attribution
- Answers are grounded in actual documentation

### 5. RAG-Based Q&A
- Retrieves top-3 relevant chunks for each query
- Uses Groq LLM (free tier) or OpenAI for generation
- Returns "The knowledge base does not contain this information." when no relevant context

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:

```bash
# Get free API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Full Pipeline

```bash
python main.py all
```

This will:
1. Scrape FastAPI documentation
2. Clean and chunk documents
3. Create embeddings and vector store
4. Generate summaries and FAQs
5. Export results to `output.md`

### 4. Ask Questions

```bash
# Single question
python main.py ask "How do I create a FastAPI route?"

# Interactive mode
python main.py interactive
```

## 📖 Usage Examples

### Ingest Documentation
```bash
# Scrape and index (first time)
python main.py ingest

# Re-index existing docs (faster)
python main.py ingest --no-scrape
```

### Generate Summaries
```bash
python main.py summarize
```

### Generate FAQs
```bash
python main.py faq --num-faqs 10
```

### Ask Questions
```bash
python main.py ask "What are path parameters in FastAPI?"
python main.py ask "How do I validate request data?"
python main.py ask "How do I run a FastAPI server?"
```

### Check Status
```bash
python main.py status
```

### Export Outputs
```bash
python main.py export --output results.md
```

## 📊 Sample Outputs

### Document Summary

> **FastAPI** is a modern, fast (high-performance) Python web framework for building APIs based on standard Python type hints. It offers automatic data validation, serialization, and interactive documentation (Swagger UI and ReDoc). FastAPI leverages Python's async capabilities for high performance comparable to NodeJS and Go. Key features include path and query parameter handling, request body parsing with Pydantic models, and automatic API documentation generation.

### FAQ Examples

**Q1: How do I create a FastAPI route?**
> Define a function and decorate it with `@app.get("/path")` or other HTTP method decorators like `@app.post()`, `@app.put()`, etc.
> *Source: First Steps*

**Q2: How do I start a FastAPI server?**
> Use `uvicorn main:app --reload` where `main` is your Python file name and `app` is the FastAPI instance.
> *Source: First Steps*

**Q3: What are path parameters?**
> Path parameters are dynamic parts of the URL path declared with curly braces, like `/items/{item_id}`. They are automatically parsed and passed to your function.
> *Source: Path Parameters*

**Q4: How do I validate query parameters?**
> Use `Query()` from FastAPI to add validation rules like `min_length`, `max_length`, and `regex` patterns.
> *Source: Query Parameters Validation*

### Query Example

**Q: How do I create a POST endpoint with request body?**

> To create a POST endpoint with a request body in FastAPI:
>
> 1. Import `BaseModel` from Pydantic
> 2. Define a model class with your data fields
> 3. Use the model as a function parameter
>
> ```python
> from fastapi import FastAPI
> from pydantic import BaseModel
>
> class Item(BaseModel):
>     name: str
>     price: float
>
> app = FastAPI()
>
> @app.post("/items/")
> async def create_item(item: Item):
>     return item
> ```
>
> *Sources: Request Body, First Steps*

## 🛠️ Tools Summary

| Component | Tool |
|-----------|------|
| Web Scraping | `requests`, `BeautifulSoup` |
| Text Processing | Custom Python |
| Chunking | Custom (500 words, 50 overlap) |
| Embeddings | `SentenceTransformers` (all-MiniLM-L6-v2) |
| Vector DB | `FAISS` (IndexFlatIP) |
| LLM | Groq (llama-3.1-8b-instant) / OpenAI |
| RAG | Custom pipeline |

## ⚠️ Limitations

- **No UI** - Console-based interface only
- **Limited Scope** - Only FastAPI tutorial documentation (10 pages)
- **Static Index** - Requires manual re-indexing for updates
- **Single Language** - English documentation only
- **API Dependency** - Requires Groq or OpenAI API key for LLM features

## 🔮 Future Improvements

1. **Web UI** - Add Streamlit or Gradio interface
2. **More Documentation** - Support Django, React, and other docs
3. **Auto-Refresh** - Scheduled re-indexing for doc updates
4. **Multi-Language** - Support translated documentation
5. **Caching** - Cache LLM responses for common queries
6. **Evaluation** - Add retrieval and answer quality metrics
7. **Hybrid Search** - Combine semantic + keyword search

## 📝 License

MIT License - Free for educational and commercial use.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Built for the Enterprise Knowledge Assistant Project**
