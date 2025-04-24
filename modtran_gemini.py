import streamlit as st
import os
import subprocess
import pdfplumber
from lxml import etree
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from keybert import KeyBERT
from sentence_transformers import CrossEncoder
import google.generativeai as genai
from typing import List
from langchain_core.language_models import BaseLanguageModel

import google.generativeai as genai


class GeminiLLM(BaseLanguageModel):
    def __init__(self, model_name="models/gemini-1.5-pro-latest", api_key=None):
        self.api_key = api_key or st.secrets["GOOGLE_API_KEY"]
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def _call(self, prompt, stop=None):
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self):
        return "custom_gemini"

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/embedding-001", api_key=None):
        api_key = "AIzaSyBIfGJRoet_wzzYXIiWXxStkIigEOzSR2o"
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )["embedding"]


class GeminiLLM:
    def __init__(self, model_name="models/gemini-1.5-pro-latest", api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def predict(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

vectorstore_global = None

def load_environment():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_file, document_name):
    text_chunks = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_chunks.append({"text": text, "page": i + 1, "document": document_name})
    return text_chunks

def get_uploaded_text(uploaded_files):
    raw_text = []
    for uploaded_file in uploaded_files:
        document_name = uploaded_file.name
        if document_name.endswith(".pdf"):
            text_chunks = extract_text_from_pdf(uploaded_file, document_name)
            raw_text.extend(text_chunks)
        elif uploaded_file.name.endswith((".html", ".htm")):
            soup = BeautifulSoup(uploaded_file.getvalue(), 'lxml')
            raw_text.append({"text": soup.get_text(), "page": None, "document": document_name})
        elif uploaded_file.name.endswith((".txt")):
            content = uploaded_file.getvalue().decode("utf-8")
            raw_text.append({"text": content, "page": None, "document": document_name})
    return raw_text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    final_chunks = []
    for chunk in raw_text:
        for split_text in splitter.split_text(chunk["text"]):
            final_chunks.append({"text": split_text, "page": chunk["page"], "document": chunk["document"]})
    return final_chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("text_chunks is empty. Cannot initialize FAISS vectorstore.")

    embeddings = GeminiEmbeddings()
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"page": chunk["page"], "document": chunk["document"]} for chunk in text_chunks]
    
    return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

def set_global_vectorstore(vectorstore):
    global vectorstore_global
    vectorstore_global = vectorstore

kw_model = KeyBERT()

def self_reasoning(query, context):
    llm = GeminiLLM()
    reasoning_prompt = f"""
    You are an AI assistant that analyzes the context provided to answer the user's query comprehensively and clearly. 
    Answer in a concise, factual way using the terminology from the context. Avoid extra explanation unless explicitly asked.
    YOU MUST mention the page number. 
    ### Example 1:
    **Question:** What is the purpose of the MODTRAN GUI?
    **Context:**
    [Page 10 of the docuemnt] The MODTRAN GUI helps users set parameters and visualize the model's output.
    **Answer:** The MODTRAN GUI assists users in parameter setup and output visualization. You can find the answer at Page 10 of the document provided.

    ### Example 2:
    **Question:** How do you run MODTRAN on Linux? Answer with page number. 
    **Context:**
    [Page 15 of the docuemnt] On Linux systems, MODTRAN can be run using the `mod6c` binary via terminal.
    **Answer:** Use the `mod6c` binary via terminal. (Page 15 of the document)

    ### Now answer:
    **Question:** {query}
    **Context:**
    {context}

    **Answer:**
    """
    return llm.predict(reasoning_prompt)

def faiss_search_with_keywords(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
    refined_query = " ".join([keyword[0] for keyword in keywords])
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(refined_query)
    context= '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content}" for doc in docs])
    return self_reasoning(query, context)

def faiss_search_with_reasoning(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(query)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in reranked_docs[:5]]
    context = '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content.strip()}" for doc in top_docs])
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

def initialize_chatbot_agent():
    llm = GeminiLLM()  # <-- Gemini instead of OpenAI
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [faiss_keyword_tool, faiss_reasoning_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent

def handle_user_query(query):
    # Same routing logic as in evaluation.py
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("Vectorstore is not initialized.")
    
    if "how" in query.lower():
        context = faiss_search_with_reasoning(query)
    else:
        context = faiss_search_with_keywords(query)
    return self_reasoning(query, context)
def main():
    load_environment()

    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with MODTRAN Documents :satellite:")
    user_question = st.text_input("Ask a question about your uploaded files:")

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload PDF, HTML, or MODTRAN output files:", accept_multiple_files=True)
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_uploaded_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(text_chunks)  
                set_global_vectorstore(st.session_state.vectorstore)         
                st.session_state.chat_ready = True
                st.success("Files processed successfully!")

    if st.session_state.chat_ready and user_question:
        # Restore the global vectorstore reference
        set_global_vectorstore(st.session_state.vectorstore)
        response = handle_user_query(user_question)
        st.session_state.chat_history.append({"user": user_question, "bot": response})

    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    load_environment()
    main()
