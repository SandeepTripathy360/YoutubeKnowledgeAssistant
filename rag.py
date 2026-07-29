
from urllib.parse import urlparse, parse_qs
from supadata import Supadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

sessions = {}

from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY")



from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=GEMINI_API_KEY
)
supadata = Supadata(api_key=SUPADATA_API_KEY)


def extract_video_id(url: str):

    if "youtube.com" in url:

        return parse_qs(
            urlparse(url).query
        )["v"][0]

    if "youtu.be" in url:

        return url.split("/")[-1]

    return url


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def process_video(url: str, session_id: str):

    video_id = extract_video_id(url)

    try:
        transcript = supadata.transcript(
            url=url,
            text=True,
            mode="auto"
        )

        transcript_text = transcript.content

    except Exception as e:
        return {
            "status": "error",
            "message": f"Transcript could not be fetched: {e}"
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents(
        [transcript_text]
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"./chroma_db/{session_id}"
    )

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5
        }
    )

    chat_history = ChatMessageHistory()

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given the chat history and latest user question,
                rewrite the question as a standalone question.

                Do not answer.

                Only rewrite if needed.
                """
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            ("human", "{question}")
        ]
    )

    rewriter_chain = (
        rewrite_prompt
        | llm
        | StrOutputParser()
    )

    retrieval_chain = (
        {
            "question":
            lambda x: x["question"],

            "chat_history":
            lambda x: x["chat_history"],

            "context":
            (
                rewriter_chain
                | base_retriever
                | RunnableLambda(format_docs)
            )
        }
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.

                Answer ONLY from the supplied context.

                If the user asks for:
                - a summary
                - what the video is about
                - main topics
                - key points

                then generate a concise summary using the retrieved context.

                If information is unavailable say:

                "I could not find that information in the transcript."

                Context:
                {context}
                """
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            ("human", "{question}")
        ]
    )
    rag_chain = (
        retrieval_chain
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    sessions[session_id] = {
        "rag_chain": rag_chain,
        "chat_history": chat_history,
        "video_id": video_id
    }

    return {
        "status": "success",
        "video_id": video_id
    }

def ask_question(
    question: str,
    session_id: str
):

    if session_id not in sessions:

        return (
            "Please process a YouTube video first."
        )

    rag_chain = sessions[
        session_id
    ]["rag_chain"]

    chat_history = sessions[
        session_id
    ]["chat_history"]

    answer = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history.messages
        }
    )

    chat_history.add_user_message(
        question
    )

    chat_history.add_ai_message(
        answer
    )

    return answer

