from django.core.management.base import BaseCommand
from django_celery_beat.models import PeriodicTask, CrontabSchedule
from users.models import User
import json
import uuid

class Command(BaseCommand):
    help = 'Schedule periodic create_and_upload_video and vector DB update tasks for all users'

    def handle(self, *args, **kwargs):
        schedule, _ = CrontabSchedule.objects.get_or_create(hour='*/8', minute='0')
        task_id = str(uuid.uuid4())

        for user in User.objects.all():
            # Schedule video creation + upload
            PeriodicTask.objects.update_or_create(
                name=f"Create Upload Video {user.id}",
                defaults={
                    "task": "api.tasks.generate_and_upload_youtube_short_task",
                    "crontab": schedule,
                    "args": json.dumps([str(user.id), task_id])
                }
            )

            # Schedule vector DB update
            PeriodicTask.objects.update_or_create(
                name=f"Update VectorDB {user.id}",
                defaults={
                    "task": "api.tasks.update_vectordb_from_youtube",
                    "crontab": schedule,
                    "args": json.dumps([str(user.id)])
                }
            )

        self.stdout.write(self.style.SUCCESS("✅ Scheduled all periodic tasks for all users!"))
