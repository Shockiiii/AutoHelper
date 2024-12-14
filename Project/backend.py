import openai
import pickle
import faiss
import numpy as np
import base64
import os
from config import OPENAI_API_KEY, GPT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION, MAX_TOKENS

# OpenAI API Key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load Doc - 修改為從上一層目錄讀取
index = faiss.read_index("faiss_index.bin")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

def get_embedding(text):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=EMBEDDING_MODEL).data[0].embedding

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
        model=GPT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content
    return f"⚠️ No relevant document found. Suggested answer based on AI knowledge:\n\n{answer}"

def query_repair_documents(car_model: str, question: str):
    query_text = f"Car model: {car_model}\nQuestion: {question}"
    
    # 使用固定維度
    query_embedding = np.array(get_embedding(query_text), dtype='float32')
    query_embedding = query_embedding.reshape(1, EMBEDDING_DIMENSION)
    
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
        model=GPT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content

    # give back result
    most_relevant_doc = retrieved_docs[0]

    return {
        "answer": answer,
        "relevant_page": most_relevant_doc
    }

def identify_part_from_image(car_model: str, image_data: bytes) -> str:
    # 将二进制数据编码为base64字符串
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # 根据官方文档示例，将图片以 data URL 的形式嵌入
    data_url = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Car model: {car_model}. What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    },
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model= GPT_MODEL,
        messages=messages,
    )

    # 返回模型对图像内容的描述
    return response.choices[0].message.content