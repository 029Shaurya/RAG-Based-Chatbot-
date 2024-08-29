from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import Ollama

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Step 1 - setup OCI Generative AI llm
# use default authN method API-key
llm = Ollama(model='llama2')

# Step 2 - Create a Prompt
base_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explains in steps."
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

load_dotenv()
embeddings = OllamaEmbeddings()
choice = (int)(input("1. Upload pdf file  2. Upload Youtube video link"))
# Define your documents
if choice == 1:
    loader = PyPDFLoader('CN-dsa.pdf')
    docs = loader.load_and_split()

    # Initialize vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Use the vector store as needed
    retriever = vectorstore.as_retriever()

    # Step 3 - Create a memory to remember our chat with the llm
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

    # Step 4 - Function to include retrieved documents in the prompt
    def get_retrieved_documents(question):
        retrieved_docs = retriever.get_relevant_documents(question)
        docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return docs_text

    # Step 5 - Modify prompt to include retrieved documents
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

    # Example usage
    question = "What are the main findings in the document?"
    conversation = create_conversation_chain(question)
    response = conversation.run(question)
    print(response)


if choice == 2:
    from pytube import YouTube
    from youtube_transcript_api import YouTubeTranscriptApi
    from fpdf import FPDF
    import os

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

    # Example usage
    video_url = 'https://www.youtube.com/watch?v=UF8uR6Z6KLc'
    transcript = get_youtube_transcript(video_url)
    save_transcript_to_pdf(transcript)

    loader = PyPDFLoader('transcript.pdf')
    docs = loader.load_and_split()

    # Initialize vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Use the vector store as needed
    retriever = vectorstore.as_retriever()

    # Step 3 - Create a memory to remember our chat with the llm
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    summary_memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

    # Step 4 - Function to include retrieved documents in the prompt
    def get_retrieved_documents(question):
        retrieved_docs = retriever.get_relevant_documents(question)
        docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return docs_text

    # Step 5 - Modify prompt to include retrieved documents
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

    # Example usage
    question = "What are the main findings in the document?"
    conversation = create_conversation_chain(question)
    response = conversation.run(question)
    print(response)

