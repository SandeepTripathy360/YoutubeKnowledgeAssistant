import streamlit as st
import requests
import uuid

API_URL = "https://youtube-knowledge-backend-677j.onrender.com"

st.set_page_config(
    page_title="YouTube Knowledge Assistant",
    page_icon="🎥",
    layout="wide"
)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

with st.sidebar:

    st.title("🎥 YouTube RAG")

    st.markdown("## Features")

    st.markdown("""
-  YouTube Transcript Extraction
-  Recursive Text Splitting
-  HuggingFace Embeddings
-  Chroma Vector Database
-  MMR Retrieval
-  Gemini 2.5 Flash
-  Conversational Memory
-  History-Aware Question Rewriting
-  FastAPI Backend
-  Streamlit Frontend
""")

    st.divider()

    st.markdown("## 📌 Workflow")

    st.markdown("""
YouTube Video

⬇️

Transcript

⬇️

Chunking

⬇️

Embeddings

⬇️

ChromaDB

⬇️

Retriever

⬇️

Gemini

⬇️

Answer
""")

    st.divider()

    if st.session_state.video_processed:
        st.success("✅ Video Processed")
    else:
        st.warning("⌛ Waiting for Video")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    if st.button("Process New Video"):

        st.session_state.messages = []
        st.session_state.video_processed = False
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()

st.title("🎥 YouTube Knowledge Assistant")

st.caption(
    "Chat with any YouTube video using Retrieval-Augmented Generation (RAG)."
)


st.markdown(
    """
Paste a YouTube URL, process its transcript,
and ask questions about the video.
"""
)

youtube_url = st.text_input(
    "📺 Enter YouTube URL"
)


if st.button("Process Video"):

    if not youtube_url:

        st.warning("Please enter a YouTube URL.")

    else:

        with st.spinner(
            "Fetching transcript and building knowledge base..."
        ):

            response = requests.post(
                f"{API_URL}/process_video",
                json={
                    "url": youtube_url,
                    "session_id": st.session_state.session_id
                }
            )

            result = response.json()

            if result.get("success"):

                st.session_state.video_processed = True

                st.success(
                    "✅ Video processed successfully! You can now start chatting."
                )

            else:

                st.error(
                    result.get(
                        "error",
                        "Unknown error occurred."
                    )
                )

if st.session_state.video_processed:

    st.divider()

    st.subheader("💬 Chat with the Video")

    # Display previous chat

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask anything about the video..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)

        with st.spinner("Gemini is thinking..."):

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id
                }
            )

            result = response.json()

            if result.get("success"):

                answer = result["answer"]

            else:

                answer = result.get(
                    "error",
                    "Unknown error."
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

else:

    st.info(
        "👈 Process a YouTube video to begin chatting."
    )

st.caption(
    "Built with ❤️ using LangChain • ChromaDB • Gemini • FastAPI • Streamlit"
)