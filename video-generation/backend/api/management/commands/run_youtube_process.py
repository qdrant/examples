from django.core.management.base import BaseCommand
from users.models import User
from api.tasks import process_youtube_channel

class Command(BaseCommand):
    help = 'Manually trigger process_youtube_channel task for a given user ID'

    def add_arguments(self, parser):
        parser.add_argument('user_id', type=str)

    def handle(self, *args, **options):
        user = User.objects.get(id=options['user_id'])
        channel_id = user.youtube_token.channel_id
        process_youtube_channel.delay(channel_id, str(user.id))
        self.stdout.write(self.style.SUCCESS(f"✅ Task triggered for user {user.email}"))
