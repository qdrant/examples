# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt update && apt install -y ffmpeg gcc libffi-dev

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app and script
COPY . .
COPY django.sh /app/django.sh
RUN chmod +x /app/django.sh

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["/app/django.sh"]
