from fastapi import FastAPI
from pydantic import BaseModel

from rag import (
    process_video,
    ask_question
)

app = FastAPI(
    title="YouTube Knowledge Assistant",
    description="YouTube RAG Chatbot using LangChain, Chroma, Ollama and FastAPI",
    version="1.0.0"
)

class VideoRequest(BaseModel):
    url: str
    session_id: str


class QuestionRequest(BaseModel):
    question: str
    session_id: str

@app.get("/")
def home():
    return {
        "message": "YouTube Knowledge Assistant API Running"
    }

@app.post("/process_video")
def process_video_endpoint(
    request: VideoRequest
):

    try:

        result = process_video(
            request.url,
            request.session_id
        )

        return {
            "success": True,
            "message": "Video processed successfully",
            "video_id": result["video_id"]
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

@app.post("/ask")
def ask_question_endpoint(
    request: QuestionRequest
):

    try:

        answer = ask_question(
            request.question,
            request.session_id
        )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }