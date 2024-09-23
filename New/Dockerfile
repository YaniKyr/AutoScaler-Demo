FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
    supervisor \
    python3 \
    python3-pip \
    golang

# Create necessary directories
RUN mkdir -p /var/log/supervisor

# Copy the supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set the working directory
WORKDIR /app

COPY go.mod go.sum ./


# Copy application files
COPY main.py requirements.txt main.go ./

RUN apt-get install -y python3.12-venv
# Install Python dependencies
RUN python3 -m venv /app/venv

# Activate the virtual environment and install Python dependencies
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install -r requirements.txt

# Build the Go client
RUN go build -o go-client main.go

# Expose the port for the Python server
EXPOSE 8081

# Start supervisord
CMD ["/usr/bin/supervisord"]
