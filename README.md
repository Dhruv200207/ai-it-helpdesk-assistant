# AI IT Helpdesk Assistant
 🚀 Live Demo: Try the AI IT Helpdesk Assistant
  https://ai-it-appdesk-assistant-xiaaylh9zrlgjwa8sawds2.streamlit.app/

An AI-powered IT troubleshooting assistant built using Python, Streamlit, LangChain, Groq LLM, FAISS, and Retrieval-Augmented Generation (RAG).

## Features

- AI-powered IT troubleshooting
- RAG-based answers from knowledge-base documents
- Semantic search using FAISS
- Groq LLM integration
- Typo tolerance for common spelling mistakes and IT acronyms
- Linux, SQL, and SSH troubleshooting knowledge base
- PDF and TXT document support
- Source references for answers
- SQLite chat history
- User feedback and analytics dashboard

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- Hugging Face Sentence Transformers
- FAISS
- SQLite
- SymSpellPy
- RapidFuzz

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dhruv200207/ai-it-helpdesk-assistant.git
cd ai-it-helpdesk-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and add your Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
RAG_MAX_L2_DISTANCE=1.25
```

Get your API key from https://console.groq.com/keys

Never upload your actual API key to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Knowledge Base

The project includes troubleshooting documents for:

- Linux
- SQL
- SSH

Additional supported documents can be added to the knowledge base.

## Author

**Dhruv Sharma**

GitHub: https://github.com/Dhruv200207

## License

No license has been added yet.