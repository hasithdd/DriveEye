FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

CMD ["main.py"]