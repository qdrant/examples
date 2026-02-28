from django.core.management.base import BaseCommand
from api.youtube_utils import ensure_qdrant_collection

class Command(BaseCommand):
    help = "Create Qdrant collection if it doesn't exist"

    def handle(self, *args, **kwargs):
        ensure_qdrant_collection()
        self.stdout.write(self.style.SUCCESS("✅ Qdrant collection checked/created."))
