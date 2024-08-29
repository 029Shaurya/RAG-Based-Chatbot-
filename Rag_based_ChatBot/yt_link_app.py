import streamlit as st
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import os
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm = Ollama(model='llama2')

embeddings = OllamaEmbeddings()

def get_youtube_transcript(video_url):
    # Extract video ID from URL
    yt = YouTube(video_url)
    video_id = yt.video_id

    # Fetch the transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)

def save_transcript_to_pdf(transcript, filename='transcript.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Set margins
    margin_left = 10
    margin_right = 10
    pdf.set_left_margin(margin_left)
    pdf.set_right_margin(margin_right)
    
    # Set width for text wrapping
    page_width = pdf.w - 2 * margin_left

    # Split the transcript into lines that fit on the PDF page
    pdf.set_font_size(12)
    pdf.multi_cell(page_width, 10, transcript)

    pdf.output(filename)

def get_retrieved_documents(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return docs_text

def create_conversation_chain(question):
    retrieved_docs = get_retrieved_documents(question)
    full_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot who explains in steps."
            ),
            SystemMessagePromptTemplate.from_template(
                f"Here are some part of video transcript to help you answer:\n{retrieved_docs}"
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    conversation = LLMChain(llm=llm, prompt=full_prompt, verbose=True, memory=memory)
    return conversation

# Streamlit app
st.set_page_config(page_title="Conversational Chatbot for your Youtube video", layout="wide")

# st.title("PDF Uploader and Loader")

# File uploader widget
youtube_url = st.text_input("Enter YouTube URL:")

if youtube_url:
    # Save the uploaded file to a temporary location
    transcript = get_youtube_transcript(youtube_url)
    save_transcript_to_pdf(transcript)
    
    loader = PyPDFLoader("transcript.pdf")
    docs = loader.load_and_split()

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

    st.markdown("""
        <style>
            body {
                background-color: #2e2e2e;
                color: #f5f5f5;
            }
            .main {
                background-color: #2e2e2e;
                padding: 20px;
            }
            .chatbox {
                background-color: #444444;
                color: #f5f5f5;
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }
            .chat-history {
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                background-color: #333333;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .user-input {
                margin-top: 10px;
            }
            .stTextInput > div > input {
                background-color: #333333;
                color: #f5f5f5;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            .stButton button {
                border-radius: 10px;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            h1 {
                color: pink;
                font-size: 2.5em;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>Conversational Chatbot for YouTube Video</h1>", unsafe_allow_html=True)


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chatbox"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chatbox"><strong>Bot:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    question = st.text_input("Ask a question:", key="question", placeholder="Type your question here...")

    if st.button("Submit"):
        if question:
            conversation = create_conversation_chain(question)
            response = conversation.run(question)
            st.session_state.chat_history.append({"question": question, "response": response})
            st.experimental_rerun()  # Rerun the app to update the chat history
