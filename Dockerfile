FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3", "main.py"]
