FROM nvcr.io/nvidia/l4t-ml:r36.4.0

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

CMD ["main.py"]