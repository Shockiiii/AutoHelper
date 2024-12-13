import openai
import os
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import time
import concurrent.futures
from typing import List, Dict
import backoff
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION, MAX_TOKENS, PDF_DIR, IMAGE_DIR, PROJECT_DIR, MAX_WORKERS, BATCH_SIZE
import re

# OpenAI API Key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 創建目錄
os.makedirs(IMAGE_DIR, exist_ok=True)

# 添加 API 重試和延遲裝飾器
@backoff.on_exception(backoff.expo, 
                     (openai.RateLimitError, openai.APIError),
                     max_tries=5)
def get_embedding_with_retry(text: str) -> np.ndarray:
    """帶重試機制的 embedding 函數"""
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return np.array(response.data[0].embedding, dtype='float32')
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise

def chunk_text(text: str) -> List[str]:
    max_tokens = MAX_TOKENS
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        # 如果段落太長，進一步分割
        if len(para) > max_tokens:
            # 按句子分割
            sentences = para.split('. ')
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_length + sentence_length > max_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        else:
            # 段落不太長，作為整體處理
            para_length = len(para)
            if current_length + para_length > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
    
    # 添加最後一個塊
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_page(page, filename: str, page_num: int, page_image) -> List[Dict]:
    results = []
    page_text = page.extract_text() or ""
    page_text = clean_text(page_text)
    
    # 验证文本质量
    if not is_valid_content(page_text):
        print(f"Skipping page {page_num} due to invalid content")
        return results
        
    # 保存图片
    page_image_path = os.path.join(IMAGE_DIR, f"{filename}_page_{page_num}.png")
    page_image.save(page_image_path, "PNG")
    
    text_chunks = chunk_text(page_text)
    
    for chunk_idx, chunk in enumerate(text_chunks):
        if not is_valid_chunk(chunk):
            continue
            
        try:
            emb = get_embedding_with_retry(chunk)
            results.append({
                "embedding": emb,
                "document": {
                    "pdf_file": filename,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "content": chunk,
                    "image_path": page_image_path
                }
            })
        except Exception as e:
            print(f"Error processing chunk {chunk_idx} of page {page_num}: {str(e)}")
            continue
            
    return results

def is_valid_content(text: str) -> bool:
    """验证文本内容是否有效"""
    # 检查文本长度
    if len(text.strip()) < 50:
        return False
        
    # 检查可读文本比例
    readable_chars = sum(c.isalnum() or c.isspace() for c in text)
    total_chars = len(text)
    if total_chars == 0 or readable_chars / total_chars < 0.7:
        return False
        
    # 检查是否包含太多特殊字符
    special_chars = sum(not (c.isalnum() or c.isspace()) for c in text)
    if special_chars / total_chars > 0.3:
        return False
        
    return True

def is_valid_chunk(chunk: str) -> bool:
    """验证文本块是否有效"""
    # 检查最小长度
    if len(chunk.strip()) < 30:
        return False
        
    # 检查是否包含完整句子
    if not re.search(r'[.!?]', chunk):
        return False
        
    return True

def process_pdf(filename: str) -> List[Dict]:
    """處理單個PDF文件"""
    all_results = []
    pdf_path = os.path.join(PDF_DIR, filename)
    
    # 轉換PDF頁面為圖片
    pages_images = convert_from_path(pdf_path, dpi=100)
    reader = PdfReader(pdf_path)
    
    # 使用較小的 worker 數量
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_page, 
                page, 
                filename, 
                i+1, 
                pages_images[i]
            )
            for i, page in enumerate(reader.pages)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing page: {str(e)}")
                continue
    
    return all_results

def save_intermediate_results(embeddings, documents, start_idx=0):
    """增量保存結果，只保留最新版本"""
    embeddings_array = np.array(embeddings[start_idx:], dtype='float32')
    
    if start_idx == 0:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        print(f"Creating new index with {len(embeddings_array)} vectors")
    else:
        index_path = os.path.join(PROJECT_DIR, "faiss_index.bin")
        if os.path.exists(index_path):
            print(f"Loading existing index and adding {len(embeddings_array)} new vectors")
            index = faiss.read_index(index_path)
        else:
            print("Previous index not found, creating new one")
            index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    
    # 添加新向量
    if len(embeddings_array) > 0:
        index.add(embeddings_array)
    
    # 保存最新版本
    faiss.write_index(index, os.path.join(PROJECT_DIR, "faiss_index.bin"))
    with open(os.path.join(PROJECT_DIR, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)
    
    print(f"Total vectors in index: {index.ntotal}")
    return index.ntotal

def clean_text(text: str) -> str:
    """清理提取的文本内容"""
    # 移除特殊编码字符
    text = re.sub(r'/[A-Z0-9]+/', ' ', text)
    # 移除多余空白
    text = ' '.join(text.split())
    # 移除无意义的短字符串
    text = '\n'.join(line for line in text.split('\n') if len(line.strip()) > 10)
    return text

# 主處理循環
documents = []
embeddings = []
batch_size = BATCH_SIZE  # 每50頁保存一次
last_save_idx = 0

for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        print(f"\nProcessing PDF: {filename}")
        start_time = time.time()
        
        results = process_pdf(filename)
        
        # 收集結果
        for result in results:
            embeddings.append(result["embedding"])
            documents.append(result["document"])
            
            # 每處理一定數量的頁面就保存一次
            current_total = len(documents)
            if current_total % batch_size == 0:
                last_save_idx = save_intermediate_results(
                    embeddings, 
                    documents, 
                    start_idx=last_save_idx
                )
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Completed processing {filename}")
        print(f"Processing time: {processing_time:.2f} seconds")

# 最後保存一次
if len(embeddings) > last_save_idx:
    save_intermediate_results(
        embeddings, 
        documents, 
        start_idx=last_save_idx
    )