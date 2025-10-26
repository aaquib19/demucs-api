# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies needed for audio processing and building
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY app.py .

# Expose the port the app runs on.
# Render will automatically map its $PORT to this exposed port.
EXPOSE 5000

# The command to run the application.
# Your app.py already correctly uses the $PORT environment variable.
CMD ["python", "app.py"]