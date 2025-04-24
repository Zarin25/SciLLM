import streamlit as st
import os
import subprocess
import pdfplumber
from lxml import etree
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from keybert import KeyBERT
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
import sys
import asyncio

# Windows fix for asyncio compatibility
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize global variable
vectorstore_global = None

# Load OpenAI API key
def load_environment():
    load_dotenv()
    
# PDF to XML Conversion
def convert_pdf_to_xml(pdf_file, xml_path):
    os.makedirs("temp", exist_ok=True)
    pdf_path = os.path.join("temp", pdf_file.name)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_file.getbuffer())
    subprocess.run(["pdftohtml", "-xml", pdf_path, xml_path], check=True)
    return xml_path

# Extract text from XML
def extract_text_from_xml(xml_path, document_name):
    from lxml import etree
    tree = etree.parse(xml_path)
    text_chunks = []
    for page in tree.xpath("//page"):
        page_num = int(page.get("number", 0))
        texts = [text.text for text in page.xpath('.//text') if text.text]
        combined_text = '\n'.join(texts)
        text_chunks.append({"text": combined_text, "page": page_num, "document": document_name})
    return text_chunks

# Process uploaded files
def get_uploaded_text(uploaded_files):
    raw_text = []
    print(f"Total uploaded files: {len(uploaded_files)}")
    for uploaded_file in uploaded_files:
        document_name = uploaded_file.name
        if document_name.endswith(".pdf"):
            xml_path = os.path.join("temp", document_name.replace(".pdf", ".xml"))
            text_chunks = extract_text_from_xml(convert_pdf_to_xml(uploaded_file, xml_path), document_name)
            raw_text.extend(text_chunks)
        elif uploaded_file.name.endswith((".html", ".htm")):
            soup = BeautifulSoup(uploaded_file.getvalue(), 'lxml')
            raw_text.append({"text": soup.get_text(), "page": None, "document": document_name})
        elif uploaded_file.name.endswith((".txt")):
            content = uploaded_file.getvalue().decode("utf-8") 
            raw_text.append({"text": content, "page": None, "document": document_name})
    return raw_text

# Text Chunking
def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    final_chunks = []
    for chunk in raw_text:
        for split_text in splitter.split_text(chunk["text"]):
            final_chunks.append({"text": split_text, "page": chunk["page"], "document": chunk["document"]})
    return final_chunks

# Vectorstore Initialization
def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("text_chunks is empty. Cannot initialize FAISS vectorstore.")

    #model_name = "BAAI/bge-large-en-v1.5"
    #encode_kwargs = {'normalize_embeddings': True}
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
    )
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"page": chunk["page"], "document": chunk["document"]} for chunk in text_chunks]

    return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

def set_global_vectorstore(vectorstore):
    global vectorstore_global
    vectorstore_global = vectorstore

kw_model = KeyBERT()

def faiss_search_with_keywords(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    
    # Extract keywords from the query
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
    refined_query = " ".join([keyword[0] for keyword in keywords])

    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(refined_query)
    
    return '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content}" for doc in docs])

def self_reasoning(query, context):
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    #llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
    reasoning_prompt = f"""
    You are an AI assistant that analyzes the context provided to answer the user's query comprehensively and clearly. 
    Answer in a concise, factual way using the terminology from the context. Avoid extra explanation unless explicitly asked.
    If asked for the page number,YOU MUST mention the page number. 
    ### Example 1:
    **Question:** What is the purpose of the MODTRAN GUI?
    **Context:**
    [Page 10 of the docuemnt] The MODTRAN GUI helps users set parameters and visualize the model's output.
    **Answer:** The MODTRAN GUI assists users in parameter setup and output visualization. You can find the answer at Page 10 of the document provided.

    ### Example 2:
    **Question:** How do you run MODTRAN on Linux? Answer with page number. 
    **Context:**
    [Page 15 of the docuemnt] On Linux systems, MODTRAN can be run using the `mod6c` binary via terminal.
    **Answer:** Use the `mod6c` binary via terminal. (Page 15)

    ### Now answer:
    **Question:** {query}
    **Context:**
    {context}

    **Answer:**
    """
    response = llm.predict(reasoning_prompt)
    return response

def faiss_search_with_reasoning(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(query)
    
    # Rerank using cross-encoder
    #pairs = [(query, doc.page_content) for doc in docs]
    #scores = reranker.predict(pairs)
    #reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    #top_docs = [doc for _, doc in reranked_docs[:5]]

    context = '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content.strip()}" for doc in docs])
    return self_reasoning(query, context)


faiss_keyword_tool = Tool(
    name="FAISS Keyword Search",
    func=faiss_search_with_keywords,
    description="Searches FAISS with a keyword-based approach to retrieve context."
)

faiss_reasoning_tool = Tool(
    name="FAISS Reasoning Search",
    func=faiss_search_with_reasoning,
    description="Searches FAISS with detailed reasoning to retrieve context."
)

# Agent Initialization
def initialize_chatbot_agent():
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    #llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [faiss_keyword_tool, faiss_reasoning_tool]
    agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        memory=memory,
        verbose=False,
        handle_parsing_errors=True)
    return agent

# Query Handler
def handle_user_query(query, agent):
    response = agent.run(query)
    return response

# Main Streamlit App
def main():
    global vectorstore_global
    load_environment()

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with MODTRAN Documents :satellite:")
    user_question = st.text_input("Ask a question about your uploaded files:")

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload PDF, HTML, or MODTRAN output files:", accept_multiple_files=True)
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_uploaded_text(uploaded_files)
                print(f"Total text chunks: {len(raw_text)}")
                if raw_text:
                    print("Example chunk:", raw_text[0])
                text_chunks = get_text_chunks(raw_text)
                vectorstore_global = get_vectorstore(text_chunks)
                st.session_state.agent = initialize_chatbot_agent()
                st.success("Files processed successfully!")

    if st.session_state.agent and user_question:
        response = handle_user_query(user_question, st.session_state.agent)
        st.session_state.chat_history.append({"user": user_question, "bot": response})

    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    load_environment()
    main()