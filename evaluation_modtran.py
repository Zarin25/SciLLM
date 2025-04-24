import os, io, re
import pandas as pd
from sklearn.metrics import accuracy_score
from bert_score import score as bert_score
import google.generativeai as genai

from modtran_gemini import (
    handle_user_query,
    initialize_chatbot_agent,
    get_uploaded_text,
    get_text_chunks,
    get_vectorstore,
    set_global_vectorstore,
    self_reasoning,
    faiss_search_with_keywords,
    faiss_search_with_reasoning
)

from langchain_openai import ChatOpenAI

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

# Load CSV dataset (ensure columns are 'question', 'answer' with no extra spaces)
df = pd.read_csv("modtran_dataset.csv")
df.columns = df.columns.str.strip()  # Strip whitespace from column names

# Load the MODTRAN user manual
with open("MODTRAN 6 User's Manual.pdf", "rb") as f:
    file_obj = io.BytesIO(f.read())
    file_obj.name = "MODTRAN 6 User's Manual.pdf"
    uploaded_files = [file_obj]

# Document processing
raw_text = get_uploaded_text(uploaded_files)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
set_global_vectorstore(vectorstore)
llm = GeminiLLM()

# Direct retrieval + answer generation
def direct_llm_rag_response(question):
    from modtran_gemini import vectorstore_global  
    if vectorstore_global is None:
        raise ValueError("Vectorstore is not initialized.")

    # Retrieve relevant documents
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(question)

    # Build a simple prompt with raw context
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
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
    **Question:** {question}
    **Context:**
    {context}

    **Answer:**
    """
    return llm.predict(prompt)

# Predict answers
df["predicted"] = df["question"].apply(direct_llm_rag_response)

# Clean up answers
true_answers = df["answer"].str.lower().str.strip()
pred_answers = df["predicted"].str.lower().str.strip()

# Normalize answers
def normalize_text(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return ' '.join(s.split())

normalized_preds = [normalize_text(p) for p in pred_answers]
normalized_refs = [normalize_text(r) for r in true_answers]

# Token-level F1
def compute_f1(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def manual_tool_routing(question):
    if "how" in question.lower():
        context = faiss_search_with_reasoning(question)
    else:
        context = faiss_search_with_keywords(question)
    return self_reasoning(question, context)

# Create predictions using different strategies
df["agent_predicted"] = df["question"].apply(manual_tool_routing)
df["keyword_predicted"] = df["question"].apply(faiss_search_with_keywords)
df["reasoning_predicted"] = df["question"].apply(faiss_search_with_reasoning)

refs = df["answer"].str.lower().str.strip()

for col in ["agent_predicted", "keyword_predicted", "reasoning_predicted"]:
    preds = df[col].str.lower().str.strip()
    normalized_preds = [normalize_text(p) for p in preds]
    normalized_refs = [normalize_text(r) for r in refs]

    em = sum([int(p == r) for p, r in zip(normalized_preds, normalized_refs)]) / len(refs)
    f1 = sum([compute_f1(p, r) for p, r in zip(normalized_preds, normalized_refs)]) / len(refs)
    P, R, F1_bert = bert_score(preds.tolist(), refs.tolist(), lang="en", verbose=True)
    bert_f1 = F1_bert.mean().item()

    print(f"\n🔹 Evaluation for: {col}")
    print(f" - Exact Match: {em:.3f}")
    print(f" - F1 Score: {f1:.3f}")
    print(f" - BERTScore F1: {bert_f1:.3f}")

    df[f"{col}_bert_f1"] = F1_bert.numpy()