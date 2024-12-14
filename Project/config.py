import streamlit as st

# 配置参数从 Streamlit Secrets 加载
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GPT_MODEL = st.secrets.get("GPT_MODEL", "gpt-4o")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(st.secrets.get("EMBEDDING_DIMENSION", 3072))
MAX_TOKENS = int(st.secrets.get("MAX_TOKENS", 500))

PDF_DIR = st.secrets.get("PDF_DIR", "pdfs")
IMAGE_DIR = st.secrets.get("IMAGE_DIR", "page_images")
PROJECT_DIR = st.secrets.get("PROJECT_DIR", "Project")

MAX_WORKERS = int(st.secrets.get("MAX_WORKERS", 4))
BATCH_SIZE = int(st.secrets.get("BATCH_SIZE", 50))
