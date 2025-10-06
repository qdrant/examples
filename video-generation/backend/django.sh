#!/bin/bash
echo "Starting Migrations..."
python manage.py migrate
echo ====================================

echo "Starting Server..."
python manage.py runserver 0.0.0.0:8000
echo "Starting Migrations..."
python manage.py migrate
python manage.py schedule_tasks
echo ====================================

echo "Starting Server..."
python manage.py runserver 0.0.0.0:8000