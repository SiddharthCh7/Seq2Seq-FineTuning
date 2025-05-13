FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py"]
