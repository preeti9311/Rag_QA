
import uuid
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
)

model = init_chat_model(
    "google_genai:gemini-3.6-flash",
    max_retries=10,
    timeout=120,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []


def document_process(path):
    try:
        # Load PDF pages
        loader = PyPDFLoader(path)
        pages = loader.load()

        if not pages:
            raise ValueError(
                "This PDF contains no readable pages."
            )

        # Keep pages that contain readable text
        readable_pages = [
            page for page in pages
            if page.page_content and page.page_content.strip()
        ]

        if not readable_pages:
            raise ValueError(
                "No readable text found. This PDF may be scanned "
                "or image-based and may require OCR."
            )

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(readable_pages)

        if not chunks:
            raise ValueError(
                "Could not extract text chunks from this PDF."
            )

        # Create vector database
        collection_name = "pdf_" + uuid.uuid4().hex

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name=collection_name,
        )

        # Update state only after successful processing
        st.session_state.vector_db = vector_db
        st.session_state.messages = []
        st.session_state.memory = []

    except ValueError:
        raise

    except Exception as e:
        raise RuntimeError(
            "Could not process this PDF. Please upload a valid, "
            "text-readable PDF and try again."
        ) from e


def extract_answer(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                if (
                    block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return str(content)


# Improvement 1: Rewrite follow-up questions
def rewrite_query(query, history):
    if not history:
        return query

    history_text = "\n".join(
        f"{item['role'].capitalize()}: {item['content']}"
        for item in history
    )

    prompt = f"""
You are a query rewriting assistant.

Rewrite the user's latest question as a standalone question
using the conversation history when necessary.

Rules:
- Resolve references like he, she, it, they, and that.
- Keep the original meaning unchanged.
- If the question is already clear, keep it essentially unchanged.
- Return only the rewritten question, without explanation.
- Do not answer the question.

Conversation history:
{history_text}

Latest question:
{query}

Standalone question:
"""

    response = model.invoke(prompt)
    rewritten_query = extract_answer(response).strip()

    return rewritten_query if rewritten_query else query


st.title("📚 Chat with Your PDF 🤖")
st.caption("Ask questions about your PDF and continue the conversation.")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
)

# Improvement 2: PDF validation and upload error handling
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("uploaded_hash") != file_hash:
        try:
            with st.spinner("Validating and processing your PDF..."):
                path = Path("uploaded_document.pdf")
                path.write_bytes(file_bytes)

                document_process(str(path))

                # Save hash only after successful processing
                st.session_state.uploaded_hash = file_hash

            st.success("PDF validated and processed successfully!")

        except ValueError as e:
            st.error(f"Invalid PDF: {e}")

        except RuntimeError as e:
            st.error(str(e))

        except Exception:
            st.error(
                "Something went wrong while processing the PDF. "
                "Please try another PDF."
            )

# Clear conversation without deleting the PDF vector database
if st.session_state.vector_db is not None:
    if st.button("🗑️ Clear chat memory"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()

# Show previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if st.session_state.vector_db is not None:
    query = st.chat_input("Ask a question about your PDF...")

    if query:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get recent conversation history
                    history = st.session_state.memory[-10:]

                    # Rewrite follow-up question before retrieval
                    search_query = rewrite_query(query, history)

                    # Retrieve relevant PDF chunks
                    docs = st.session_state.vector_db.similarity_search(
                        search_query,
                        k=4,
                    )

                    context = "\n\n".join(
                        f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )

                    # Build conversation history for Gemini
                    history_text = "\n".join(
                        f"{item['role'].capitalize()}: {item['content']}"
                        for item in history
                    )

                    prompt = f"""
You are a helpful PDF question-answering assistant.

Use the PDF context and previous conversation to understand
the user's question, including follow-up questions.

Rules:
- Answer PDF-related questions using the provided PDF context.
- Use previous conversation to resolve references when needed.
- Do not treat previous assistant answers as proof if the
  PDF context does not support the answer.
- If the answer is not available in the PDF context,
  clearly say you don't know.
- Include source page numbers when available.
- Do not invent facts.

Previous conversation:
{history_text if history_text else "No previous conversation."}

PDF context:
{context}

Current question:
{query}

Answer:
"""

                    response = model.invoke(prompt)
                    answer = extract_answer(response)

                    if not answer:
                        answer = "Sorry, I could not generate a text answer."

                    st.markdown(answer)

                    # Save conversation for display and future memory
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state.memory.extend(
                        [
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": answer},
                        ]
                    )

                    # Keep only the latest 10 conversation messages
                    st.session_state.memory = (
                        st.session_state.memory[-10:]
                    )

                except Exception as e:
                    st.error(f"Could not generate an answer: {e}")

                    # Remove the unanswered user message
                    if (
                        st.session_state.messages
                        and st.session_state.messages[-1]["role"] == "user"
                    ):
                        st.session_state.messages.pop()
