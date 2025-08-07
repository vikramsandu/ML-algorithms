# Use a base Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
CMD ["python", "main.py"]
