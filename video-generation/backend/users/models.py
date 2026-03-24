from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
import uuid

from google.auth import default
from .utils import encrypt_value, decrypt_value

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email must be set")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra_fields)

class User(AbstractBaseUser, PermissionsMixin):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    # Optional: storing OpenAI, Replicate, ElevenLabs keys per user
    openai_api_key = models.TextField(null=True, blank=True)
    replicate_api_key = models.TextField(null=True, blank=True)
    elevenlabs_api_key = models.TextField(null=True, blank=True)
    openai_model = models.CharField(max_length=255, default='gpt-4o')
    elevenlabs_voice_id = models.CharField(max_length=255, null=True, blank=True)
    audience = models.TextField(null=True, blank=True)
    flux_model = models.CharField(max_length=255, default="black-forest-labs/flux-schnell")

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    def save(self, *args, **kwargs):
        # Encrypt keys before saving
        if self.openai_api_key and not self.openai_api_key.startswith('gAAAA'):
            self.openai_api_key = encrypt_value(self.openai_api_key)
        if self.replicate_api_key and not self.replicate_api_key.startswith('gAAAA'):
            self.replicate_api_key = encrypt_value(self.replicate_api_key)
        if self.elevenlabs_api_key and not self.elevenlabs_api_key.startswith('gAAAA'):
            self.elevenlabs_api_key = encrypt_value(self.elevenlabs_api_key)
        super().save(*args, **kwargs)

    @property
    def openai_api_key_decrypted(self):
        return decrypt_value(self.openai_api_key)

    @property
    def replicate_api_key_decrypted(self):
        return decrypt_value(self.replicate_api_key)

    @property
    def elevenlabs_api_key_decrypted(self):
        return decrypt_value(self.elevenlabs_api_key)

    def __str__(self):
        return self.email

class YouTubeToken(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="youtube_token")
    access_token = models.TextField()
    refresh_token = models.TextField()
    token_uri = models.CharField(max_length=255)
    client_id = models.CharField(max_length=255)
    client_secret = models.CharField(max_length=255)
    scopes = models.TextField()
    channel_id = models.CharField(max_length=255, null=True, blank=True)
    expiry = models.DateTimeField()


class Video(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="videos")
    title = models.CharField(max_length=255)
    description = models.TextField()
    video_url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title