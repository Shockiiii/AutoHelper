# 使用 Python 轻量级镜像
FROM python:3.9-slim

# 设置工作目录（**改为 project 目录**）
WORKDIR /app/project

# 复制整个项目
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r /app/project/requirements.txt

# 确保 `gunicorn` 可执行
RUN pip install gunicorn

# **正确启动 Gunicorn**
CMD ["gunicorn", "-b", "0.0.0.0:8000", "backend:app"]
