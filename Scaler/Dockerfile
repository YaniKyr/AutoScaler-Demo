FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    supervisor \
    python3 \
    python3-pip \
    golang

# Create necessary directories
RUN mkdir -p /var/log/supervisor

# Set the working directory
WORKDIR /app

COPY /py-scripts/requirements.txt ./

RUN apt-get install -y python3.12-venv
# Install Python dependencies
RUN python3 -m venv /app/venv

# Activate the virtual environment and install Python dependencies
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install -r requirements.txt

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY /go-scripts/go.mod /go-scripts/go.sum /go-scripts/main ./

# Copy application files
COPY /py-scripts/env.py /py-scripts/predict.py /py-scripts/offline_model.zip /py-scripts/functions.py /go-scripts/main.go  ./



# Start supervisord
CMD ["/usr/bin/supervisord"]
