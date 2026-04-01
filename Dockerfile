FROM python:3.10-slim

WORKDIR /app

# Copy the pyproject.toml we made earlier
COPY pyproject.toml .

# Install dependencies directly from pyproject.toml
RUN pip install --no-cache-dir .

# Copy the rest of your files (including the server folder)
COPY . .

ENV PYTHONPATH=/app
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]