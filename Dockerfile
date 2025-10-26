# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies
# We install build tools first, then remove them after pip install to keep the image small.
RUN apt-get update && apt-get install -y \
    # Runtime dependencies for audio processing
    ffmpeg \
    libsndfile1 \
    # Build dependencies (needed for compiling Python C extensions like 'diffq')
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY app.py .

# The app will listen on port 5000 inside the container
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]