FROM python:3.7-slim-buster
WORKDIR .
COPY . .
RUN pip install -r requirements.txt
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 app:app
EXPOSE 8080