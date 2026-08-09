<div align="center">

# 🧠 Knowledge Intelligence System

### AI-Powered Document Q&A with Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge)](https://trychroma.com)
[![AWS S3](https://img.shields.io/badge/AWS_S3-Storage-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/s3/)

<br/>

*Upload documents, ask questions in natural language, and get AI-generated answers grounded in your data — not hallucinations.*

<br/>

[Features](#-features) · [Architecture](#-architecture) · [Getting Started](#-getting-started) · [API Reference](#-api-reference) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works)

</div>

---

## 📌 Overview

**Knowledge Intelligence System** is a full-stack Retrieval-Augmented Generation (RAG) application that lets users upload PDF and text documents, index them into a vector database, and ask natural language questions that are answered by GPT-4.1 using only the context from the uploaded documents.

Unlike vanilla ChatGPT, this system **grounds every answer in your documents** — reducing hallucinations and enabling domain-specific Q&A over private data.

### Why RAG?

| Approach | Limitation |
|----------|-----------|
| **Vanilla LLM** | Can hallucinate; no access to your private data |
| **Fine-tuning** | Expensive; requires retraining for every new document |
| **RAG (this project)** | ✅ Real-time document ingestion, ✅ Grounded answers, ✅ No retraining needed |

---

## ✨ Features

- 📄 **Multi-format Document Upload** — Supports PDF and TXT files via drag-and-drop or click-to-browse
- 🔍 **Semantic Search** — Documents are embedded and indexed in ChromaDB for similarity-based retrieval
- 🤖 **Conversational AI** — Context-aware follow-up questions using chat history
- 💾 **Persistent Storage** — Documents are backed up to AWS S3; vector embeddings persist across restarts
- 🎨 **Modern Dark UI** — Glassmorphism design with animated backgrounds, micro-interactions, and responsive layout
- ⚡ **Real-time Feedback** — Upload progress, chunk counts, and inline error messages
- 🛡️ **Error Handling** — Granular try/catch blocks with structured JSON error responses
- 📊 **Logging** — Structured logging at every step for debugging and monitoring

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (UI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Upload Zone   │  │ Query Input  │  │  Chat History         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
└─────────┼─────────────────┼─────────────────────────────────────┘
          │ POST /upload    │ POST /query
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Backend (main.py)                       │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │ process_document()│    │         LLMService                │   │
│  │                    │    │  ┌────────────────────────────┐  │   │
│  │ 1. Save to disk   │    │  │ History-Aware Retriever    │  │   │
│  │ 2. Load (PDF/TXT) │    │  │ (Reformulate question)     │  │   │
│  │ 3. Chunk (1000ch) │    │  ├────────────────────────────┤  │   │
│  │ 4. Return chunks  │    │  │ Retrieval Chain            │  │   │
│  └────────┬─────────┘    │  │ (Search → Generate answer) │  │   │
│           │               │  └────────────────────────────┘  │   │
│           ▼               └──────────────┬───────────────────┘   │
│  ┌────────────────┐                      │                        │
│  │ StorageService │◄─── S3 Backup        │                        │
│  └────────────────┘                      ▼                        │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    VectorStore (ChromaDB)                   │   │
│  │  Embed via OpenAI → Store vectors → Similarity search      │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   ┌─────────────┐              ┌──────────────┐
   │   AWS S3     │              │  OpenAI API   │
   │  (Backup)    │              │  (Embeddings  │
   │              │              │   + GPT-4.1)  │
   └─────────────┘              └──────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5, CSS3, JavaScript | Responsive UI with drag-and-drop, chat interface |
| **Backend** | Flask (Python) | REST API server, request routing |
| **AI/LLM** | LangChain + OpenAI GPT-4.1-mini | RAG pipeline, prompt chaining, chat memory |
| **Vector DB** | ChromaDB | Embedding storage, similarity search |
| **Embeddings** | OpenAI `text-embedding-ada-002` | Convert text chunks → dense vectors |
| **Cloud Storage** | AWS S3 + Boto3 | Document backup and persistence |
| **Document Parsing** | PyPDFLoader, TextLoader | Extract text from PDF and TXT files |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunk documents with overlap for context |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **OpenAI API Key** — [Get one here](https://platform.openai.com/api-keys)
- **AWS Account** with S3 access (access key + secret key + bucket)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Knowledge-Intelligence-System---Langchain_RAG.git
cd Knowledge-Intelligence-System---Langchain_RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `app/` directory:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
AWS_BUCKET_NAME=your-s3-bucket-name
```

### 5. Run the Application

```bash
cd app
python main.py
```

The server starts at **http://localhost:8080**

---

## 📡 API Reference

### `GET /`

Serves the main web interface.

---

### `POST /upload`

Upload and index a document.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | A `.pdf` or `.txt` file to upload |

**Response:**

```json
{
  "message": "File uploaded and processed successfully",
  "chunks_processed": 42
}
```

**Error Responses:**

| Status | Reason |
|--------|--------|
| `400` | No file provided / empty filename / unsupported file type |
| `500` | Document processing error / S3 upload failure / indexing failure |

---

### `POST /query`

Ask a question against the indexed documents.

**Request:** `application/json`

```json
{
  "question": "What are the key findings in the report?"
}
```

**Response:**

```json
{
  "response": "The report identifies three key findings: ..."
}
```

**Error Responses:**

| Status | Reason |
|--------|--------|
| `400` | No question provided |
| `500` | LLM/retrieval error |

---

## 📁 Project Structure

```
Knowledge-Intelligence-System---Langchain_RAG/
├── app/
│   ├── main.py                 # Flask app entry point, routes, document processing
│   ├── config.py               # Environment variable loader (dotenv)
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── vectorStore.py      # ChromaDB wrapper (add documents, similarity search)
│   │   └── __init__.py
│   │
│   ├── service/
│   │   ├── llmService.py       # RAG chain (history-aware retriever + answer generation)
│   │   ├── storageService.py   # AWS S3 file upload service
│   │   └── __init__.py
│   │
│   ├── template/
│   │   └── index.html          # Main UI (upload, query, chat history)
│   │
│   ├── static/
│   │   └── style.css           # Dark glassmorphism theme, animations
│   │
│   └── .env                    # API keys (not committed to git)
│
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 How It Works

### 1. Document Ingestion Pipeline

```
Upload File → Save to Temp → Load (PDF/TXT) → Split into Chunks → Embed → Store in ChromaDB → Backup to S3
```

1. User uploads a PDF or TXT file via the web UI
2. The file is saved to a temporary directory on disk
3. **PyPDFLoader** or **TextLoader** parses the file into LangChain `Document` objects
4. **RecursiveCharacterTextSplitter** splits documents into 1000-character chunks with 200-character overlap
5. Chunks are embedded using **OpenAI's embedding model** and stored in **ChromaDB**
6. The original file is backed up to **AWS S3**

### 2. Question-Answering Pipeline (RAG)

```
User Question → Reformulate with Chat History → Retrieve Relevant Chunks → Generate Grounded Answer
```

1. User submits a natural language question
2. **History-Aware Retriever** reformulates the question using conversation history (for follow-up questions like "tell me more about *that*")
3. The reformulated question is used to **search ChromaDB** for the most relevant document chunks (similarity search)
4. Retrieved chunks are injected into a prompt as **context**
5. **GPT-4.1-mini** generates an answer grounded in the retrieved context
6. The question and answer are appended to **chat history** for multi-turn conversations

### 3. Conversational Memory

The system maintains chat history across questions within a session, enabling:
- **Coreference resolution** — "What was the second point?" refers back to a previous answer
- **Follow-up questions** — "Can you elaborate on that?" uses context from the last response
- **Multi-turn reasoning** — Each question builds on the previous conversation

---

## 🎨 UI Preview

The interface features a modern dark theme with:
- **Glassmorphism cards** with backdrop blur and subtle borders
- **Animated gradient blobs** floating in the background
- **Drag-and-drop file upload** with visual feedback
- **Chat bubble interface** showing conversation history
- **Responsive grid layout** that adapts to mobile screens
- **Micro-animations** — hover effects, fade-in transitions, loading spinners

---

## 🔐 Security Considerations

- 🔒 API keys are stored in `.env` and loaded via `python-dotenv` — never hardcoded
- 🚫 `.env` is included in `.gitignore` to prevent accidental commits
- ⚠️ The Flask development server (`debug=True`) should **not** be used in production
- 📁 Uploaded files are processed in temporary directories and cleaned up immediately

---

## 🛣 Roadmap

- [ ] Add support for more file formats (DOCX, CSV, Markdown)
- [ ] Implement user authentication and multi-tenant document isolation
- [ ] Add streaming responses for real-time answer generation
- [ ] Deploy with Gunicorn + Nginx for production readiness
- [ ] Add unit and integration tests
- [ ] Implement document management (list, delete indexed documents)
- [ ] Add source citations in AI responses (show which document chunks were used)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using LangChain, OpenAI, ChromaDB, and Flask**

</div>