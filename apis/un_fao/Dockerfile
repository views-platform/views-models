# Use official Python 3.11 image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire views-models repository
COPY . .

# Set the working directory to the API location
WORKDIR /app/apis/un_fao

# Install Python dependencies
RUN pip install --no-cache-dir \
    wandb==0.18.7 \
    fastapi==0.118.2 \
    uvicorn==0.37.0 \
    appwrite==13.3.0 \
    python-dotenv==0.9.9 \
    art==6.4 \
    pyprojroot==0.3.0 \
    joblib==1.5.2 \
    pandas==2.3.3 \
    numpy==2.3.4 \
    fastparquet==2024.11.0 \
    pyarrow==21.0.0

# Make run.sh executable
RUN chmod +x run.sh

# Expose the port
EXPOSE 8000

# Use the run.sh script as entrypoint
ENTRYPOINT ["./docker_run.sh"]