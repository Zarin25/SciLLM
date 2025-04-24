import os
import io
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from modtran_gemini import (
    handle_user_query,
    initialize_chatbot_agent,
    get_uploaded_text,
    get_text_chunks,
    get_vectorstore,
    set_global_vectorstore
)
import string
import re

load_dotenv()

# Load SQuAD dataset for benchmark
ds = load_dataset("squad", split="validation[:100]")

# Extract context, questions, and answers
contexts = [item["context"] for item in ds]
questions = [item["question"] for item in ds]
answers = [item["answers"]["text"][0] for item in ds]

# Create dataframe
df = pd.DataFrame({"context": contexts, "question": questions, "answer": answers})

# Save contexts to disk to simulate file uploads
os.makedirs("squad_contexts", exist_ok=True)
for i, context in enumerate(df["context"].unique()):
    with open(f"squad_contexts/context_{i}.txt", "w", encoding="utf-8") as f:
        f.write(context)

# Simulate file uploads (Streamlit-like file objects)
uploaded_files = []
for filename in os.listdir("squad_contexts"):
    if filename.endswith(".txt"):
        with open(os.path.join("squad_contexts", filename), "rb") as f:
            file_obj = io.BytesIO(f.read())
            file_obj.name = filename
            uploaded_files.append(file_obj)

# Initialize vectorstore and agent
raw_text = get_uploaded_text(uploaded_files)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
set_global_vectorstore(vectorstore)

agent = initialize_chatbot_agent()

# Run chatbot predictions
df["chatbot_answer"] = df["question"].apply(lambda q: handle_user_query(q, agent))

# BLEU Evaluation
references = [[ans.split()] for ans in df["answer"]]
hypotheses = [pred.split() for pred in df["chatbot_answer"]]
bleu_score = corpus_bleu(references, hypotheses)

# ROUGE Evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(ref, hyp) for ref, hyp in zip(df["answer"], df["chatbot_answer"])]
rouge1 = sum(score["rouge1"].fmeasure for score in rouge_scores) / len(rouge_scores)
rougeL = sum(score["rougeL"].fmeasure for score in rouge_scores) / len(rouge_scores)

# BERTScore Evaluation
P, R, F1 = score(df["chatbot_answer"].tolist(), df["answer"].tolist(),model_type='distilbert-base-uncased', batch_size=4,lang="en", verbose=True)
mean_precision = P.mean().item()
mean_recall = R.mean().item()
mean_f1 = F1.mean().item()

# SQuAD F1 Evaluation
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = set(gold_toks) & set(pred_toks)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

df["squad_f1_score"] = df.apply(lambda row: compute_f1(row["answer"], row["chatbot_answer"]), axis=1)
mean_squad_f1 = df["squad_f1_score"].mean()

# Print evaluation results
print(f"BLEU Score: {bleu_score:.3f}")
print(f"ROUGE-1: {rouge1:.3f}")
print(f"ROUGE-L: {rougeL:.3f}")
# BERTScore Evaluation
P, R, F1 = score(df["chatbot_answer"].tolist(), df["answer"].tolist(), lang="en")
df["BERTScore_F1"] = F1.numpy()
print(f"Mean BERTScore F1: {F1.mean().item():.3f}")
# Save evaluation results
df.to_csv('evaluation_results.csv', index=False)
