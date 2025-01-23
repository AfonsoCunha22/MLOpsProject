FROM python:3.12-slim

# Expose the port that the application listens on.
EXPOSE 8080

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY front_end/requirements_backend.txt requirements_backend.txt
COPY front_end/app.py app.py
RUN pip3 install -r requirements_backend.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
