import streamlit as st
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()


llm = Ollama(model='llama2')


# base_prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#             "You are a nice chatbot who explains in steps."
#         ),
#         HumanMessagePromptTemplate.from_template("{question}"),
#     ]
# )

embeddings = OllamaEmbeddings()


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
                f"Here are some relevant documents to help you answer:\n{retrieved_docs}"
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    conversation = LLMChain(llm=llm, prompt=full_prompt, verbose=True, memory=memory)
    return conversation

# Streamlit app
st.set_page_config(page_title="Conversational Chatbot for your pdfs", layout="wide")

# st.title("PDF Uploader and Loader")

    # File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
        # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

        # Load and split the PDF
    loader = PyPDFLoader("temp.pdf")
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
                color: #f5f5f5;
                font-size: 2.5em;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown("<h1>Conversational Chatbot for your pdf</h1>", unsafe_allow_html=True)


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