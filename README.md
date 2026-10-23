# CV RAG Assistant

Ask questions about a CV and get answers grounded in its contents. The application uses section-tagged chunks, OpenAI embeddings, local persistent Qdrant storage, metadata-filtered retrieval, and a LangChain chat model.

## Architecture

1. **CV extraction:** `rag.py` reads text from PDF and TXT files using pypdf or UTF-8. It also extracts PDF hyperlink annotations and visible URLs, then adds formatted GitHub, LinkedIn, and project links to the matching section chunks.
2. **Section metadata:** headings are extracted from the CV before chunking. Each chunk includes its normalized section, original `section_header`, an expanded `section_header_text` for matching, source filename, chunk index, CV ID, and stable chunk ID. If headings are missing, lightweight text cues assign best-effort section labels.
3. **Chunking:** each section is split into overlapping 180-word passages with 35-word overlap.
4. **Embeddings and storage:** LangChain's `OpenAIEmbeddings` uses `text-embedding-3-small`. Vectors and metadata are stored in a persistent local Qdrant database at `qdrant_storage/`.
5. **Metadata prefilter and retrieval:** all chunk metadata is loaded into memory. The question and each unique section header are embedded with the same model, cosine similarity scores each header, and only headers above `HEADER_SIMILARITY_THRESHOLD` (default `0.18`) are passed to Qdrant as a metadata filter before vector similarity search. The sidebar lets the user adjust the threshold or further restrict the eligible section labels.
6. **Answering:** LangChain's `ChatOpenAI` sends the retrieved passages and question to the configured OpenAI model. The prompt requires answers to stay within the passages and report missing information. The UI shows passing header scores and retrieved passages with section and source filename.

The local Qdrant database persists CV text, vectors, and metadata on disk. Remove `qdrant_storage/` to delete all locally indexed CVs. API keys are read from `.env` or entered in the sidebar and are not saved by the app.

## Models and configuration

The default chat model is `gpt-6-luna`, selected for cost-sensitive chat usage. The embedding model is `text-embedding-3-small`, a compact OpenAI embedding model suitable for semantic search. Model names can be changed in `.env`.

The project contains an empty `.env` template. Add your key to `OPENAI_API_KEY`, then adjust the model settings if needed:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-6-luna
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
QDRANT_PATH=qdrant_storage
HEADER_SIMILARITY_THRESHOLD=0.18
```

Never commit API keys. `.env` and the local Qdrant database are excluded by `.gitignore`.

## Start on Windows

Double-click `start.bat`. It creates `.venv` if needed, installs dependencies, starts Streamlit, and opens the UI in your browser. The first dependency installation requires an internet connection. You can also run it from PowerShell:

```powershell
cd "D:\curt\Gen AI\task2\CV-Based-RAG-Application"
.\start.bat
```

## Manual setup

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a text-based PDF or TXT CV. Scanned PDFs need OCR before upload. The first time a CV is indexed, its chunks are embedded using the configured OpenAI embedding model and written to local Qdrant. Later app starts reuse that index when the same CV and embedding model are uploaded.

## Naive RAG scope

Retrieval uses a metadata header similarity gate followed by one semantic vector search. It does not use query rewriting, query expansion, HyDE, reranking, or contextual compression. Section labels are inferred from common CV headings and simple cues, so review the displayed source passages when a label seems incorrect.
