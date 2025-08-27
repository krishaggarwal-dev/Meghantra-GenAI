# Meghantra – GenAI Document Assistant

**📌 Overview**

Meghantra is a GenAI-powered document assistant that allows users to upload files and query them intelligently. It is designed for academic and learning use cases by enabling structured question answering and automatic quiz generation from documents.

The project leverages LangChain, DeepSeek-R1 (via Ollama), HuggingFace embeddings, and ChromaDB to provide retrieval-augmented generation (RAG) capabilities, combined with a simple Streamlit UI.

**✨ Features**

📂 Upload and process PDF / TXT documents

🔍 Ask questions from uploaded documents (context-aware answers)

📝 Generate quizzes (MCQ format with options, correct answer, and explanation)

💾 Vector database storage using ChromaDB for efficient retrieval

🎯 Designed for academic, research, and learning purposes

**🛠️ Tech Stack**

Language: Python

Frameworks & Libraries:

Streamlit
 → interactive UI

LangChain
 → chaining & orchestration

PyPDF2
 → PDF text extraction

ChromaDB
 → vector database

SentenceTransformers
 → HuggingFace embeddings (all-MiniLM-L6-v2)

Model: DeepSeek-R1:1.5B
 via Ollama

**📐 Architecture**

Document Upload → Users upload PDFs/TXT files

Preprocessing → Documents are split into chunks & embedded

Vector Store → Embeddings stored in ChromaDB

Query/Quiz Mode

Q&A Mode → Retrieve relevant chunks → DeepSeek-R1 generates structured answers

Quiz Mode → Generate MCQs with 4 options, correct answer & explanation

**🚀 Future Enhancements**

Add quiz difficulty levels (Easy, Moderate, Hard)

Support for multiple quiz types (Fill in the Blanks, Descriptive, etc.)

Improved summarization of long documents

Multi-user session support

📄 License

This project is licensed under the MIT License.
