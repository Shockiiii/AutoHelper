import openai
import os
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# OpenAI API Key
client = openai.OpenAI(

api_key="",



)


EMBED_MODEL = "text-embedding-3-large"
PDF_DIR = "pdfs"
IMAGE_DIR = "page_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype='float32')

documents = []
embeddings = []


for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        
        # pdf to img
        pages = convert_from_path(pdf_path, dpi=100)
        
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text is None:
                page_text = ""
            page_text = page_text.strip()
            
            # save img
            page_image_path = os.path.join(IMAGE_DIR, f"{filename}_page_{i+1}.png")
            pages[i].save(page_image_path, "PNG")
            
            if page_text:
                emb = get_embedding(page_text)
                embeddings.append(emb)
                documents.append({
                    "pdf_file": filename,
                    "page_number": i+1,
                    "content": page_text,
                    "image_path": page_image_path
                })

if len(embeddings) == 0:
    print("Folder empty")
    exit(1)

embeddings = np.array(embeddings, dtype='float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")

with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("data ready, create faiss_index.bin 和 documents.pkl。")