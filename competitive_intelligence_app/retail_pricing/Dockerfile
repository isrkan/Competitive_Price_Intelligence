# Use the official Python base image
FROM python:3.9.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app code
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the entry point for the container to run the Flask app
CMD ["python", "app.py"]