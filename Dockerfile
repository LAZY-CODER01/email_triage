FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment validation variables
ENV PYTHONPATH=/app

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the FastAPI server to keep the Space alive
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]