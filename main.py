from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import (
    process_video,
    ask_question
)

app = FastAPI(
    title="YouTube Knowledge Assistant",
    description="YouTube RAG Chatbot using LangChain, Chroma, Gemini and FastAPI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # We'll tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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