# Use an official lightweight Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary dependencies
RUN pip install --upgrade pip
RUN pip install Flask tensorflow pillow

# Expose port 5000 for the Flask server
EXPOSE 5000

# Set the default command to run the inference server
CMD ["python", "server.py"]