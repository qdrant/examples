from rest_framework.views import APIView
from rest_framework.response import Response
from celery.result import AsyncResult
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
import uuid
import os
from .tasks import generate_and_upload_youtube_short_task
import logging
from api.redis_client import r

class UserAPIKeysView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        return Response({
            "openai_api_key": user.openai_api_key_decrypted,
            "replicate_api_key": user.replicate_api_key_decrypted,
            "elevenlabs_api_key": user.elevenlabs_api_key_decrypted,
        })

class TestTaskView(APIView):
    def post(self, request):
        task = test_celery_task.delay(2, 3)
        return Response({"task_id": task.id})

class TaskStatusView(APIView):
    def get(self, request, task_id):
        result = AsyncResult(task_id)
        return Response({
            "state": result.state,
            "result": str(result.result) if result.ready() else None,
        }, status=status.HTTP_200_OK)

class GenerateAndUploadShortView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        try:
            result = generate_and_upload_youtube_short(user.id)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def generate_and_upload_youtube_short(user_id):
    task_id = str(uuid.uuid4())
    log_path = os.path.join("shorts", f"make_short_{task_id}.log")
    os.makedirs("shorts", exist_ok=True)
    logger = logging.getLogger(f"make_short_{task_id}")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Received user request for short video generation")
    logger.info(f"Task ID: {task_id}")
    r.hset(f"task:{task_id}", mapping={
            "status": "queued",
            "type": "generate_and_upload_youtube_short"
        })

    task = generate_and_upload_youtube_short_task.delay(user_id, task_id)
    logger.info(f"user_id = {user_id}, type = {type(user_id)}")
    logger.info(f"task_id = {task_id}, type = {type(task_id)}")

    logger.info(f"Dispatched Celery task {task.id}")
    logger.removeHandler(handler)
    handler.close()
    return {"status": "queued", "task_id": task.id}
