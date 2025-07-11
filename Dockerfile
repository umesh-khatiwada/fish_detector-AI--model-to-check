FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app
# Add this line to set pip's cache and temp directories to a writable location
ENV TMPDIR=/tmp PIP_CACHE_DIR=/tmp/.pip_cache

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --cache-dir=/tmp/.pip_cache -r requirements.txt

COPY . .

CMD ["python3", "batch_label.py"]
