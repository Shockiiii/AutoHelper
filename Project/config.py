import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 3072))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 500))

# 文件路径配置
PDF_DIR = os.getenv('PDF_DIR', 'pdfs')
IMAGE_DIR = os.getenv('IMAGE_DIR', 'page_images')
PROJECT_DIR = os.getenv('PROJECT_DIR', 'Project')

# 处理配置
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
BATCH_SIZE = 50