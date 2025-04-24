from datasets import load_dataset
import pandas as pd
import os, io
from bert_score import score
from sklearn.metrics import accuracy_score, f1_score
from modtran import (
    handle_user_query,
    initialize_chatbot_agent,
    get_uploaded_text,
    get_text_chunks,
    get_vectorstore,
    set_global_vectorstore,
)

# Load SciQ
ds = load_dataset("sciq", split="validation[:100]")

# Extract supports as context, question, and correct answer
contexts = [item["support"] for item in ds]
questions = [item["question"] for item in ds]
answers = [item["correct_answer"] for item in ds]
predictions = []

# Create dataframe
df = pd.DataFrame({"context": contexts, "question": questions, "answer": answers})

# Save contexts to disk to simulate file uploads
os.makedirs("sciq_contexts", exist_ok=True)
for i, context in enumerate(df["context"].unique()):
    with open(f"sciq_contexts/context_{i}.txt", "w", encoding="utf-8") as f:
        f.write(context)

# Simulate file uploads
uploaded_files = []
for filename in os.listdir("sciq_contexts"):
    if filename.endswith(".txt"):
        with open(os.path.join("sciq_contexts", filename), "rb") as f:
            file_obj = io.BytesIO(f.read())
            file_obj.name = filename
            uploaded_files.append(file_obj)

print("Total uploaded files:", len(uploaded_files))

# Vectorstore pipeline
raw_text = get_uploaded_text(uploaded_files)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
set_global_vectorstore(vectorstore)

# Initialize chatbot agent
agent = initialize_chatbot_agent()


# Predict answers
df["chatbot_answer"] = df["question"].apply(lambda q: handle_user_query(q, agent))

# BERTScore Evaluation
P, R, F1 = score(df["chatbot_answer"].tolist(), df["answer"].tolist(), lang="en")
df["BERTScore_F1"] = F1.numpy()

print(f"Mean BERTScore F1: {F1.mean().item():.3f}")
for q in questions:
    pred = handle_user_query(q, agent)
    predictions.append(pred)

# Compute Accuracy
acc = accuracy_score(answers, predictions)
print(f"Accuracy: {acc:.3f}")

# Compute F1 (macro average - good for open-ended QA)
f1 = f1_score(answers, predictions, average='macro')
print(f"F1 Score (macro): {f1:.3f}")

# Save results
df.to_csv("sciq_evaluation_results.csv", index=False)
