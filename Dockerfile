# Use a lightweight official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV H2O_JAVA_OPTIONS="-Xmx2g"

# Set working directory
WORKDIR /app

# Install Java (required by h2o)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code and model
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
