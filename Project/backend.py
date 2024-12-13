import openai
import pickle
import faiss
import numpy as np
import os

# OpenAI API Key
client = openai.OpenAI(

api_key="",



)

GPT_4O_MODEL = "gpt-4o"

# Load Doc
index = faiss.read_index("faiss_index.bin")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)


def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def ask_gpt_knowledge_base(car_model: str, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an automotive expert assisting with car repairs."
        },
        {
            "role": "user",
            "content": f"Car model: {car_model}\nQuestion: {question}\n**Note:** This answer is generated based on general model knowledge, not from the provided manuals."
        },
    ]
    response = client.chat.completions.create(
        model=GPT_4O_MODEL,
        messages=messages,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return f"⚠️ No relevant document found. Suggested answer based on AI knowledge:\n\n{answer}"

def query_repair_documents(car_model: str, question: str):
   
    query_text = f"Car model: {car_model}\nQuestion: {question}"
   
    query_embedding = np.array(get_embedding(query_text), dtype='float32').reshape(1, -1)

    k = 3
    D, I = index.search(query_embedding, k)

    retrieved_docs = []
    for idx in I[0]:
        if idx == -1:
            continue
        doc = documents[idx]
        retrieved_docs.append(doc)

    if not retrieved_docs:
        # not find
        return {
            "answer": ask_gpt_knowledge_base(car_model, question),
            "relevant_page": None
        }

    # found and pass
    combined_context = "\n\n".join([f"PDF: {d['pdf_file']} (page {d['page_number']}):\n{d['content']}" for d in retrieved_docs])

    messages = [
        {
            "role": "system",
            "content": "You are a car repair assistant referencing provided PDF manuals."
        },
        {
            "role": "user",
            "content": f"Based on the following repair manual pages, answer the question:\n\n{combined_context}\n\nCar model: {car_model}\nQuestion: {question}"
        },
    ]

    response = client.chat.completions.create(
        model=GPT_4O_MODEL,
        messages=messages,
        max_tokens=500,
    )
    answer = response.choices[0].message.content

    # give back result
    most_relevant_doc = retrieved_docs[0]

    return {
        "answer": answer,
        "relevant_page": most_relevant_doc
    }
