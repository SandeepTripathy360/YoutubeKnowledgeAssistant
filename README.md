# YouTube Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to interact with YouTube videos through natural language. The application extracts a video's transcript, stores it in a vector database, and answers user queries using Google's Gemini model.

## Features

- Process YouTube videos using their URL
- Fetch transcripts automatically
- Store transcript embeddings in ChromaDB
- Semantic search using HuggingFace embeddings
- Context-aware question answering with Gemini
- Session-based conversation history
- Support for English and Hindi transcripts
- FastAPI backend with Streamlit frontend

## Tech Stack

**Frontend**
- Streamlit

**Backend**
- FastAPI
- Uvicorn

**LLM**
- Google Gemini 2.5 Flash

**Embeddings**
- BAAI/bge-base-en-v1.5

**Vector Database**
- ChromaDB

**Framework**
- LangChain

## Project Structure

```
YoutubeKnowledgeAssistant/
│
├── app_show.py          # Streamlit frontend
├── main.py              # FastAPI application
├── rag.py               # RAG pipeline
├── requirements.txt
├── .gitignore
├── .env
│
├── chroma_db/
└── __pycache__/
```

## How It Works

1. The user enters a YouTube video URL.
2. The transcript is retrieved using the YouTube Transcript API.
3. The transcript is split into smaller chunks.
4. HuggingFace embeddings are generated for each chunk.
5. The embeddings are stored in ChromaDB.
6. Relevant chunks are retrieved based on the user's query.
7. Gemini generates an answer using the retrieved context.
8. The response is displayed in the Streamlit interface.

## Installation

Clone the repository

```bash
git clone https://github.com/SandeepTripathy360/YoutubeKnowledgeAssistant.git
cd YoutubeKnowledgeAssistant
```

Create a virtual environment

```bash
python -m venv myenv
```

Activate the environment

**Windows**

```bash
myenv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project directory.

```text
GEMINI_API_KEY=your_api_key
```

## Running the Project

Start the FastAPI server

```bash
uvicorn main:app --reload
```

In another terminal, start the Streamlit application

```bash
streamlit run app_show.py
```

The application will be available at:

```
http://localhost:8501
```

## Future Improvements

- Support videos without transcripts using speech-to-text.
- Display timestamp-based source citations.
- Support multiple videos in a single knowledge base.
- Deploy the application on cloud platforms.
- Improve retrieval with hybrid search and reranking.

## Screenshots

Add screenshots of the application here.

```
<img width="1896" height="917" alt="image" src="https://github.com/user-attachments/assets/f8bf6fda-7d95-43f5-bf6a-7f462128a804" />

```

## Author

**Sandeep Tripathy**

GitHub: https://github.com/SandeepTripathy360
