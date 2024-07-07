FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update and install necessary system packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Expose port 80 for the application
EXPOSE 80

# Set environment variable
ENV NAME World

# Run the start script when the container launches
CMD ["./run_scripts.sh"]
