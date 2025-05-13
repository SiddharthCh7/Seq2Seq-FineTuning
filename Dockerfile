FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py"]
