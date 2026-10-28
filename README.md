# 📚 PDF Chat Assistant

An AI-powered PDF chatbot that allows users to upload a PDF document and ask questions about its content using natural language.

## ✨ Features

- 📄 Upload and process PDF documents
- 🔍 Extract text and split it into chunks
- 🧠 Generate embeddings using Ollama (`nomic-embed-text`)
- 🗄️ Store and retrieve document chunks using ChromaDB
- 🤖 Generate context-aware answers using Google Gemini
- 💬 Support conversational follow-up questions
- 🛡️ Validate PDFs and handle processing errors
- 🧹 Clear chat history without re-uploading the document

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- Ollama
- ChromaDB

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Never commit your `.env` file or expose your API key publicly.

### 5. Set up Ollama

Install Ollama and download the embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running.

### 6. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚀 Deployment

This application can be deployed as a Streamlit web service on Render. Configure the required environment variables and ensure the embedding model is accessible to the deployed application.

**Note:** Ollama must be configured separately for hosted deployment; it is not automatically available on Render.

## 🔐 Security

- Store API keys in environment variables.
- Do not commit `.env` files or secrets.
- Avoid uploading confidential documents to an untrusted deployment.

## 👨‍💻 Future Improvements

- Display reliable source citations for retrieved PDF passages.
- Support OCR for scanned PDFs.
- Improve retrieval quality and document management.

