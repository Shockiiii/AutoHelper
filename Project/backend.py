import openai
import pickle
import faiss
import numpy as np
import base64
import os
from config import OPENAI_API_KEY, GPT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION, MAX_TOKENS

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)  # Enable Cross-Origin Requests

# Set OpenAI API Key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load FAISS index and document data (ensure file paths are correct)
faiss_index = faiss.read_index("./faiss_index.bin")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Set a distance threshold (adjust based on your embedding metrics)
THRESHOLD = 1.3

def get_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

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
    return answer

def is_doc_relevant(car_model: str, pdf_file: str) -> bool:
    """
    使用 AI 判断给定 PDF 文件名是否与选定的车辆相关。
    由于文件命名可能不规则，此处调用 GPT 模型判断是否可能属于该车辆的维修手册。
    返回 True 表示相关，False 表示不相关。
    """
    prompt = (
        f"Selected vehicle: {car_model}\n"
        f"PDF file name: {pdf_file}\n"
        "Does the PDF file name suggest that this document is related to the repair manual for the selected vehicle? "
        "Answer with 'Yes' or 'No'."
    )
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print("Error in is_doc_relevant:", e)
        return False

def query_repair_documents(car_model: str, question: str, history: list = None, image_description: str = None) -> dict:
    """
    根据车辆、问题、（可选）对话历史和图像描述，从 PDF 文档中检索相关信息，并生成回答。
    如果过滤后没有找到与所选车辆相关的维修手册，则在回答最上方提示用户当前没有找到具体的文件，
    但仍调用 GPT 生成基于通用知识的回答。
    """
    # 1. 构造查询文本，计算嵌入以检索 PDF 文档
    query_text = f"Car model: {car_model}\nQuestion: {question}"
    query_embedding = np.array(get_embedding(query_text), dtype='float32')
    query_embedding = query_embedding.reshape(1, EMBEDDING_DIMENSION)
    
    k = 3
    distances, indices = faiss_index.search(query_embedding, k)
    best_distance = distances[0][0]
    print("Best matching distance:", best_distance)
    
    # 2. 检索相关文档
    retrieved_docs = []
    for idx in indices[0]:
        if idx == -1:
            continue
        doc = documents[idx]
        retrieved_docs.append(doc)
    
    # 3. 使用 AI 判断每个检索到的文档是否与所选车辆相关
    filtered_docs = []
    for doc in retrieved_docs:
        pdf_file = doc.get("pdf_file", "")
        if is_doc_relevant(car_model, pdf_file):
            filtered_docs.append(doc)
    
    # 4. 如果过滤后没有找到相关文档，则提示用户，并调用 GPT 生成通用知识回答
    if not filtered_docs:
        note = "Note: No specific repair manual was found for the selected vehicle.\n"
        general_answer = ask_gpt_knowledge_base(car_model, question)
        answer = note + general_answer
        return {
            "answer": answer,
            "relevant_page": None,
            "history": history if history else []
        }
    
    # 5. 合并相关文档内容为上下文字符串
    pdf_context = "\n\n".join([
        f"PDF: {d['pdf_file']} (page {d['page_number']}):\n{d['content']}" for d in filtered_docs
    ])
    
    # 6. 构造最终查询内容
    let_combined_query = f"Car model: {car_model}\nQuestion: {question}"
    if image_description:
        let_combined_query = f"Image recognition result: {image_description}\n" + let_combined_query
    let_final_query = let_combined_query + "\n\n" + pdf_context

    # 7. 构造消息历史：系统提示 + (如有历史，则追加历史) + 当前用户消息
    messages = [
        {
            "role": "system",
            "content": "You are a car repair assistant referencing provided PDF manuals."
        }
    ]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": let_final_query
    })
    
    # 8. 调用 ChatGPT API生成回答
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content
    
    # 9. 返回最相关文档信息（取第一个）
    most_relevant_doc = filtered_docs[0]
    pdf_file_name = most_relevant_doc["pdf_file"]
    pdf_url = f"/pdfs/{pdf_file_name}"
    most_relevant_doc["pdf_url"] = pdf_url
    
    return {
        "answer": answer,
        "relevant_page": most_relevant_doc,
        "history": messages
    }

def identify_part_from_image(car_model: str, image_data: bytes) -> str:
    # 将图片二进制数据编码为 base64 字符串
    base64_image = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Car model: {car_model}. What is shown in this image?",
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
        model=GPT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

# -------------------------------
# API Routes
# -------------------------------
@app.route("/api/query_repair_documents", methods=["POST"])
def api_query_repair_documents():
    data = request.get_json()
    car_model = data.get("car_model")
    question = data.get("question")
    history = data.get("history")           # Conversation history (list)
    image_description = data.get("image_description")  # Optional image description
    if not car_model or not question:
        return jsonify({"error": "Missing car_model or question"}), 400
    result = query_repair_documents(car_model, question, history, image_description)
    return jsonify(result)

@app.route("/api/identify_part_from_image", methods=["POST"])
def api_identify_part_from_image():
    car_model = request.form.get("car_model")
    if not car_model:
        return jsonify({"error": "Missing car_model"}), 400
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    image_file = request.files["image"]
    image_data = image_file.read()
    description = identify_part_from_image(car_model, image_data)
    return jsonify({"description": description})

@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    # Build the absolute path for the pdfs folder (located one level above the project folder)
    pdf_dir = os.path.abspath(os.path.join(app.root_path, "..", "pdfs"))
    print("Available files in pdfs:", os.listdir(pdf_dir))
    full_path = os.path.join(pdf_dir, filename)
    print("Requesting PDF:", full_path)
    if not os.path.exists(full_path):
        print("File does not exist!")
    else:
        print("File exists, serving file.")
    return send_from_directory(pdf_dir, filename)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
