# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements_backend.txt /app/requirements-requirements_backend.txt
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy the application files
COPY . /app/

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
