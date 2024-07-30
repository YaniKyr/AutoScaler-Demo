FROM python:3.9
WORKDIR /app
COPY main.py requirements.txt  ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python","main.py"]